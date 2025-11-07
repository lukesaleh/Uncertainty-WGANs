import os
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

# Import torchmetrics for FID and precision/recall
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Import models
from models import Generator_Uncertainty, Generator_CNN_Uncertainty, Critic, Critic_Simple

# Move data loading into a function to avoid global variables causing issues with multiprocessing in windows
def get_data_loaders(batch_size):
    print("Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, persistent_workers=True)
    print(f"Train samples: {train_len}, Val samples: {val_len}")
    
    return train_loader, val_loader

# Gradient penalty for WGAN-GP during training
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interp = epsilon * real + (1 - epsilon) * fake
    interp_scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=interp_scores,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Train/val helpers
def gen_rand_noise(n, z_dim, device):
    return torch.randn(n, z_dim, device=device)

def critic_accuracy(real_scores, fake_scores):
    # real>fake threshold
    real_correct = (real_scores > 0).float().mean().item()
    fake_correct = (fake_scores < 0).float().mean().item()
    return real_correct, fake_correct

def add_label_to_image(img_tensor, label, font_size=10):
    """Add a label (0 or 1) to the top-left corner of an image tensor"""
    # Convert tensor to PIL Image
    img = F.to_pil_image(img_tensor)
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw white background rectangle for label
    text = str(label)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle([0, 0, text_width + 4, text_height + 4], fill='white')
    
    # Draw black text
    draw.text((2, 2), text, fill='black', font=font)
    
    # Convert back to tensor
    return F.to_tensor(img)

# =========================
# Training and Validation
# =========================

def run_validation(G, D, val_loader, latent_dim, device):
    D.eval()
    G.eval()
    val_critic_loss = 0.0
    val_real_acc = 0.0
    val_fake_acc = 0.0
    count = 0
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for real, _ in val_bar:
            real = real.to(device)
            z = gen_rand_noise(real.size(0), latent_dim, device)
            fake = G(z)

            real_scores = D(real)
            fake_scores = D(fake)

            # WGAN critic loss 
            loss_D = (fake_scores.mean() - real_scores.mean()).item()
            val_critic_loss += loss_D

            r_acc, f_acc = critic_accuracy(real_scores, fake_scores)
            val_real_acc += r_acc
            val_fake_acc += f_acc
            count += 1
            
            # Update FID - convert to uint8 [0, 255] and 3 channels for Inception
            real_imgs = ((real + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            fake_imgs = ((fake + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            
            # Convert grayscale to RGB by repeating channels
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
            
            fid.update(real_imgs, real=True)
            fid.update(fake_imgs, real=False)

    # Compute FID score
    fid_score = fid.compute().item()
    
    D.train()
    G.train()
    return (val_critic_loss / count,
            val_real_acc / count,
            val_fake_acc / count,
            fid_score)


def train(G, D, opt_G, opt_D, train_loader, val_loader, config, device):
    print("\nStarting training...")
    best_fid = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, config['num_epochs'] + 1):
        G.train()
        D.train()
        running_D_loss = 0.0
        running_G_loss = 0.0
        running_real_acc = 0.0
        running_fake_acc = 0.0
        batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
        for i, (real, _) in enumerate(train_bar):
            real = real.to(device)
            bsz = real.size(0)

            # Update critic n_critic times
            for _ in range(config['n_critic']):
                z = gen_rand_noise(bsz, config['latent_dim'], device)
                fake = G(z).detach()
                real_scores = D(real)
                fake_scores = D(fake)

                gp = gradient_penalty(D, real, fake, device)
                loss_D = (fake_scores.mean() - real_scores.mean()) + config['lambda_gp'] * gp

                opt_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                opt_D.step()

            z = gen_rand_noise(bsz, config['latent_dim'], device)
            fake = G(z)
            fake_scores_for_G = D(fake)

            # batch means
            with torch.no_grad():
                real_scores_for_mean = D(real)
            mu_fake = fake_scores_for_G.mean().detach()
            mu_real = real_scores_for_mean.mean().detach()

            # uncertainty = how far above avg fake you are, capped by gap
            gap = (mu_real - mu_fake).clamp(min=1e-6)
            uncertainty = (fake_scores_for_G - mu_fake).clamp(min=0.0, max=gap)

            loss_G_adv = -fake_scores_for_G.mean()
            loss_G_unc = -config['uncertainty_lambda'] * uncertainty.mean()

            loss_G = loss_G_adv + loss_G_unc

            opt_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            opt_G.step()

            # track losses and acc
            running_D_loss += loss_D.item()
            running_G_loss += loss_G.item()
            r_acc, f_acc = critic_accuracy(real_scores_for_mean, fake_scores_for_G.detach())
            running_real_acc += r_acc
            running_fake_acc += f_acc
            batches += 1
            
            # update progress bar
            train_bar.set_postfix({
                'D_loss': f'{running_D_loss/batches:.3f}',
                'G_loss': f'{running_G_loss/batches:.3f}'
            })

        # Validation 
        print("Running validation...")
        val_loss, val_racc, val_facc, fid_score = run_validation(
            G, D, val_loader, config['latent_dim'], device
        )

        print(f"Epoch [{epoch}/{config['num_epochs']}] "
              f"Train D loss: {running_D_loss/batches:.4f} | "
              f"Train G loss: {running_G_loss/batches:.4f} | "
              f"Train D real acc: {running_real_acc/batches:.3f} | "
              f"Train D fake acc: {running_fake_acc/batches:.3f} || "
              f"Val D loss: {val_loss:.4f} | Val real acc: {val_racc:.3f} | "
              f"Val fake acc: {val_facc:.3f} | FID: {fid_score:.2f}")

        # Save sample every 5 epochs
        if epoch % 5 == 0:
            print(f"Saving sample images...")
            G.eval()
            with torch.no_grad():
                sample_z = gen_rand_noise(16, config['latent_dim'], device)
                fake_imgs = G(sample_z)
                fake_imgs = (fake_imgs + 1) / 2.0
                utils.save_image(fake_imgs, os.path.join(config['sample_dir'], f"fake_epoch_{epoch}.png"), nrow=4)
            G.train()

        # Early stopping based on FID
        if fid_score < best_fid - 0.5:
            best_fid = fid_score
            epochs_no_improve = 0
            print(f"New best FID! ({fid_score:.2f}) Saving...")
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "epoch": epoch,
                "fid": fid_score
            }, os.path.join(config['model_dir'], "gan_uncertainty_fmnist_best_" + config['model_name_append'] + ".pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print(f"Early stopping at epoch {epoch} (best FID: {best_fid:.2f})")
                break
    
    print("Training complete!")

def test(G, config, device, val_loader):
    print("Loading model for testing...")
    checkpoint = torch.load(config['model_to_load'], map_location=device, weights_only=False)
    G.load_state_dict(checkpoint["G"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'fid' in checkpoint:
        print(f"Model training FID: {checkpoint['fid']:.2f}")
    
    print("\nGenerating test samples and computing metrics...")
    G.eval()
    
    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    
    num_samples = 2000
    samples_generated = 0
    
    # Collect some real images for side-by-side comparison
    real_samples = []
    
    with torch.no_grad():
        # Generate fake samples
        print("Generating fake samples...")
        while samples_generated < num_samples:
            batch_size = min(64, num_samples - samples_generated)
            z = gen_rand_noise(batch_size, config['latent_dim'], device)
            fake = G(z)
            
            # Convert to uint8 [0, 255] and RGB
            fake_imgs = ((fake + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
            
            fid.update(fake_imgs, real=False)
            inception_score.update(fake_imgs)
            samples_generated += batch_size
        
        # Get real samples
        print("Processing real samples...")
        real_processed = 0
        for real, _ in val_loader:
            real = real.to(device)
            
            # Save first 16 real images for comparison
            if len(real_samples) < 16:
                remaining = 16 - len(real_samples)
                real_samples.append(real[:remaining])
            
            # Convert to uint8 [0, 255] and RGB
            real_imgs = ((real + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            
            fid.update(real_imgs, real=True)
            real_processed += real.size(0)
            
            if real_processed >= num_samples:
                break
        
        # Concatenate real samples
        real_samples = torch.cat(real_samples, dim=0)[:16]
        
        # Generate 16 fake samples for comparison
        print("Generating comparison samples...")
        z = gen_rand_noise(16, config['latent_dim'], device)
        fake_samples = G(z)
        
        # Scale to [0, 1] for saving
        real_samples_scaled = (real_samples + 1) / 2.0
        fake_samples_scaled = (fake_samples + 1) / 2.0
        
        # Add labels to images
        real_samples_labeled = []
        fake_samples_labeled = []
        
        for i in range(16):
            real_img_labeled = add_label_to_image(real_samples_scaled[i].cpu(), label=1)
            fake_img_labeled = add_label_to_image(fake_samples_scaled[i].cpu(), label=0)
            real_samples_labeled.append(real_img_labeled)
            fake_samples_labeled.append(fake_img_labeled)
        
        real_samples_labeled = torch.stack(real_samples_labeled)
        fake_samples_labeled = torch.stack(fake_samples_labeled)
        
        # Create side-by-side comparison with labels
        comparison = torch.stack([real_samples_labeled, fake_samples_labeled], dim=1)
        comparison = comparison.view(-1, 1, 28, 28) 
        
        # Save individual sets with labels
        utils.save_image(real_samples_labeled, 
                        os.path.join(config['test_dir'], "real_samples_labeled.png"), 
                        nrow=4, normalize=False)
        utils.save_image(fake_samples_labeled, 
                        os.path.join(config['test_dir'], "fake_samples_labeled.png"), 
                        nrow=4, normalize=False)
        
        # Save without labels too
        utils.save_image(real_samples_scaled, 
                        os.path.join(config['test_dir'], "real_samples.png"), 
                        nrow=4, normalize=False)
        utils.save_image(fake_samples_scaled, 
                        os.path.join(config['test_dir'], "fake_samples.png"), 
                        nrow=4, normalize=False)
        
        # Save side-by-side comparison with labels
        utils.save_image(comparison, 
                        os.path.join(config['test_dir'], "real_vs_fake_comparison_labeled.png"), 
                        nrow=8, normalize=False)
        
        # Save side-by-side comparison without labels
        comparison_unlabeled = torch.stack([real_samples_scaled, fake_samples_scaled], dim=1)
        comparison_unlabeled = comparison_unlabeled.view(-1, 1, 28, 28)
        utils.save_image(comparison_unlabeled, 
                        os.path.join(config['test_dir'], "real_vs_fake_comparison.png"), 
                        nrow=8, normalize=False)
        
        # Create grid with labels
        # Top 2 rows = real, bottom 2 rows = fake
        grid_real = utils.make_grid(real_samples_labeled, nrow=4, normalize=False, padding=2)
        grid_fake = utils.make_grid(fake_samples_labeled, nrow=4, normalize=False, padding=2)
        
        # Stack vertically with a separator
        separator = torch.ones(3, grid_real.size(1), 5, device='cpu') * 0.5 
        grid_combined = torch.cat([grid_real, separator, grid_fake], dim=2)
        
        utils.save_image(grid_combined, 
                        os.path.join(config['test_dir'], "real_vs_fake_stacked_labeled.png"), 
                        normalize=False)
    
    # Compute metrics
    print("\nComputing final metrics...")
    fid_score = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"{'='*50}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    print(f"\nSample images saved:")
    print(f"  - Real samples (unlabeled): {os.path.join(config['test_dir'], 'real_samples.png')}")
    print(f"  - Fake samples (unlabeled): {os.path.join(config['test_dir'], 'fake_samples.png')}")
    print(f"  - Real samples (labeled): {os.path.join(config['test_dir'], 'real_samples_labeled.png')}")
    print(f"  - Fake samples (labeled): {os.path.join(config['test_dir'], 'fake_samples_labeled.png')}")
    print(f"  - Interleaved comparison (labeled): {os.path.join(config['test_dir'], 'real_vs_fake_comparison_labeled.png')}")
    print(f"  - Interleaved comparison (unlabeled): {os.path.join(config['test_dir'], 'real_vs_fake_comparison.png')}")
    print(f"  - Stacked comparison (labeled): {os.path.join(config['test_dir'], 'real_vs_fake_stacked_labeled.png')}")
    print(f"{'='*50}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # Config
    # =========================
    config = {
        'batch_size': 64,
        'latent_dim': 100,
        'num_epochs': 100,
        'lr_g': 2e-4,
        'lr_d': 5e-5,
        'beta1': 0.5,
        'beta2': 0.9,
        'lambda_gp': 10.0,
        'uncertainty_lambda': 0.5,
        'n_critic': 3, 
        'patience': 10,
        'run_test': True,
        'use_cnn_generator': True,  # Set to True to use CNN generator
        'img_height': 28,  # Image height
        'img_width': 28,   # Image width
        'img_channels': 1  # 1 for grayscale, 3 for RGB
    }

    # Set model type and create separate directories for CNN vs MLP
    if config['use_cnn_generator']:
        model_type = "cnn"
        config['model_name_append'] = "cnn"
    else:
        model_type = "mlp"
        config['model_name_append'] = "mlp"
    
    # Create separate directories for each model type
    config['model_dir'] = os.path.join(os.getcwd(), "models", "models_uncertainty", model_type)
    config['sample_dir'] = os.path.join(os.getcwd(), "results", "uncertainty", model_type, "samples")
    config['test_dir'] = os.path.join(os.getcwd(), "results", "uncertainty", model_type, "test")
    
    # Set model path to load
    config['model_to_load'] = os.path.join(config['model_dir'], f"gan_uncertainty_fmnist_best_{model_type}.pth")

    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    os.makedirs(config['test_dir'], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Model Directory: {config['model_dir']}")
    print(f"Sample Directory: {config['sample_dir']}")
    print(f"Test Directory: {config['test_dir']}")
    print(f"{'='*60}\n")

    # Get data loaders
    train_loader, val_loader = get_data_loaders(config['batch_size'])
    
    # Initialize models based on config
    if config['use_cnn_generator']:
        print(f"Using CNN-based generator for {config['img_height']}x{config['img_width']} images")
        G = Generator_CNN_Uncertainty(
            z_dim=config['latent_dim'],
            img_channels=config['img_channels'],
            img_height=config['img_height'],
            img_width=config['img_width']
        ).to(device)
        D = Critic(
            img_channels=config['img_channels'],
            img_height=config['img_height'],
            img_width=config['img_width']
        ).to(device)
    else:
        print("Using fully connected generator")
        G = Generator_Uncertainty(config['latent_dim']).to(device)
        D = Critic_Simple().to(device)
    
    print("Models initialized")

    # SEPARATE LEARNING RATES
    opt_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))
    opt_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
    
    if config['run_test']:
        test(G, config, device, val_loader)
    else:
        train(G, D, opt_G, opt_D, train_loader, val_loader, config, device)
