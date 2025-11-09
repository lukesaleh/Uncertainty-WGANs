import os
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from tqdm import tqdm
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

# Import Hugging Face datasets
from datasets import load_dataset

# Import torchmetrics for FID and precision/recall
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Import models
from models import Generator_Uncertainty, Generator_CNN_Uncertainty, Critic, Critic_Simple

# Import distributed training utilities
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Define collate_fn at module level for Windows compatibility
def collate_fn(batch):
    """Custom collate function to handle numpy arrays from HuggingFace dataset"""
    import numpy as np
    images = []
    labels = []
    
    for item in batch:
        # Handle both numpy arrays and lists
        pixel_values = item['pixel_values']
        if isinstance(pixel_values, list):
            pixel_values = np.asarray(pixel_values)
        elif not isinstance(pixel_values, np.ndarray):
            pixel_values = np.asarray(pixel_values)
        
        img = torch.from_numpy(pixel_values).float()
        images.append(img)
        labels.append(item['label'])
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_data_loaders(batch_size, img_height=256, img_width=256, val_split=0.15, use_augmentation=True, max_samples=None):
    """
    Load NIH Chest X-ray dataset from Hugging Face
    """
    print("="*70)
    print("Loading NIH Chest X-ray dataset from Hugging Face...")
    print("This may take a few minutes on first run (dataset will be cached)")
    print("="*70)

    # Determine number of processes
    num_proc = os.cpu_count() - 2 if os.cpu_count() is not None and os.cpu_count() > 2 else 1

    try:
        print("\nDownloading/Loading dataset...")
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            name="image-classification",
            split="train",
            trust_remote_code=True, num_proc=num_proc
        )
        
        print(f"Dataset loaded: {len(dataset):,} images")
        
        if max_samples is not None and max_samples < len(dataset):
            print(f"Limiting to {max_samples:,} samples for testing...")
            dataset = dataset.select(range(max_samples))
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        raise
    
    # Define transform functions (non-batched)
    def apply_train_transform(example):
        """Apply transformations to a single example"""
        image = example['image']
        
        if image.mode != 'L':
            image = image.convert('L')
        
        if use_augmentation:
            transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        tensor = transform(image)
        return {'pixel_values': tensor.numpy(), 'label': 0}
    
    def apply_eval_transform(example):
        """Apply evaluation transformations to a single example"""
        image = example['image']
        
        if image.mode != 'L':
            image = image.convert('L')
        
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        tensor = transform(image)
        return {'pixel_values': tensor.numpy(), 'label': 0}
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(0.15 * total_size)
    val_size = int(val_split * (total_size - test_size))
    train_size = total_size - val_size - test_size
    
    print(f"\nSplitting dataset:")
    print(f"  Total: {total_size:,}")
    print(f"  Train: {train_size:,} ({train_size/total_size*100:.1f}%)")
    print(f"  Val:   {val_size:,} ({val_size/total_size*100:.1f}%)")
    print(f"  Test:  {test_size:,} ({test_size/total_size*100:.1f}%)")
    
    # Split using train_test_split from datasets
    train_val_split = dataset.train_test_split(test_size=test_size, seed=42)
    train_val_dataset = train_val_split['train']
    test_dataset = train_val_split['test']
    
    val_split_ratio = val_size / (train_size + val_size)
    train_val_split2 = train_val_dataset.train_test_split(test_size=val_split_ratio, seed=42)
    train_dataset = train_val_split2['train']
    val_dataset = train_val_split2['test']
    
    # Apply transforms using map with multiple processes
    print(f"\nApplying transforms with {num_proc} processes...")
    train_dataset = train_dataset.map(
        apply_train_transform,
        remove_columns=['image'],
        desc="Transforming training data",
        num_proc=num_proc
    )
    
    val_dataset = val_dataset.map(
        apply_eval_transform,
        remove_columns=['image'],
        desc="Transforming validation data",
        num_proc=num_proc
    )
    
    test_dataset = test_dataset.map(
        apply_eval_transform,
        remove_columns=['image'],
        desc="Transforming test data",
        num_proc=num_proc
    )
    
    print(f"\n{'='*70}")
    print(f"NIH Chest X-ray Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Training Set:   {len(train_dataset):,} images")
    if use_augmentation:
        print(f"  └─ With augmentation (effective: ~{len(train_dataset)*5:,})")
    print(f"Validation Set: {len(val_dataset):,} images")
    print(f"Test Set:       {len(test_dataset):,} images")
    print(f"Total:          {total_size:,} images")
    print(f"\nImage Size:     {img_height}x{img_width}")
    print(f"{'='*70}\n")
    
    # Optimize workers for multi-GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    cpu_count = os.cpu_count() if os.cpu_count() is not None else 8
    
    # For DDP: os.cpu_count() returns total, but we get cpus-per-task via SLURM
    if dist.is_initialized():
        # With DDP, each process gets its own CPUs
        cpus_per_process = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count))
        num_workers = min(12, cpus_per_process - 2)  # Cap at 12, leave 2 for main
        print(f"\nDDP Mode: {cpus_per_process} CPUs per process")
    elif num_gpus >= 4:
        # DataParallel with 4+ GPUs
        num_workers = min(16 * num_gpus, cpu_count - 4)
    else:
        num_workers = min(8 * num_gpus, cpu_count - 2)
    
    num_workers = max(4, num_workers)  # Minimum 4 workers
    
    print(f"\nDataLoader Configuration:")
    print(f"  GPUs: {num_gpus}")
    print(f"  CPU cores available: {cpu_count}")
    print(f"  DataLoader workers: {num_workers}")
    if dist.is_initialized():
        print(f"  Mode: DistributedDataParallel")
        print(f"  Workers per process: {num_workers}")
    else:
        print(f"  Mode: {'DataParallel' if num_gpus > 1 else 'Single GPU'}")
        print(f"  Workers per GPU: {num_workers // num_gpus if num_gpus > 1 else num_workers}")
    
    persistent = True
    
    # Create samplers for DDP
    train_sampler = None
    val_sampler = None
    shuffle_train = True
    
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False  # Sampler handles shuffling
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,  # Add this
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,  # Enable for faster GPU transfer
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4  if num_gpus > 1 else 2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,  # Add this
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True if num_gpus > 1 else False,  # Enable pin_memory for validation too
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_gpus > 1 else 2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=persistent if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


# Gradient penalty for WGAN-GP during training
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    
    # Ensure epsilon is on the same device as real/fake
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device, dtype=real.dtype)
    
    # Create interpolated samples
    interp = (epsilon * real + (1 - epsilon) * fake).detach()
    interp.requires_grad_(True)
    
    # Forward pass - handle DataParallel module access
    if isinstance(critic, nn.DataParallel):
        interp_scores = critic.module(interp)
    else:
        interp_scores = critic(interp)
    
    # Compute gradients
    grads = torch.autograd.grad(
        outputs=interp_scores,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_scores, device=interp_scores.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten and compute penalty
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

def add_label_to_image(img_tensor, label, font_size=20):
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
    draw.rectangle([0, 0, text_width + 8, text_height + 8], fill='white')
    
    # Draw black text
    draw.text((4, 4), text, fill='black', font=font)
    
    # Convert back to tensor
    return F.to_tensor(img)

# =========================
# Training and Validation
# =========================

def run_validation(G, D, val_loader, latent_dim, device, max_fid_samples=5000):
    """
    Run validation with limited FID computation for speed
    """
    D.eval()
    G.eval()
    val_critic_loss = 0.0
    val_real_acc = 0.0
    val_fake_acc = 0.0
    count = 0
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Track samples processed for FID
    fid_samples_processed = 0
    compute_fid = True
    
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for real, _ in val_bar:
            real = real.to(device)
            batch_size = real.size(0)
            z = gen_rand_noise(batch_size, latent_dim, device)
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
            
            # Update FID only if under max_fid_samples
            if compute_fid and fid_samples_processed < max_fid_samples:
                # Determine how many samples to use from this batch
                samples_to_use = min(batch_size, max_fid_samples - fid_samples_processed)
                
                # Convert to uint8 [0, 255] and 3 channels for Inception
                real_imgs = ((real[:samples_to_use] + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
                fake_imgs = ((fake[:samples_to_use] + 1) / 2.0 * 255).clamp(0, 255).to(torch.uint8)
                
                # Convert grayscale to RGB by repeating channels
                real_imgs = real_imgs.repeat(1, 3, 1, 1)
                fake_imgs = fake_imgs.repeat(1, 3, 1, 1)
                
                fid.update(real_imgs, real=True)
                fid.update(fake_imgs, real=False)
                
                fid_samples_processed += samples_to_use
                
                # Stop computing FID once we reach max_fid_samples
                if fid_samples_processed >= max_fid_samples:
                    compute_fid = False
                    val_bar.set_description(f"Validation (FID: {fid_samples_processed} samples)")
            
            # Update progress
            val_bar.set_postfix({
                'D_loss': f'{val_critic_loss/count:.3f}',
                'FID_samples': fid_samples_processed
            })

    # Compute FID score
    if fid_samples_processed > 0:
        fid_score = fid.compute().item()
        print(f"FID computed on {fid_samples_processed} samples")
    else:
        fid_score = float('inf')
        print("Warning: Not enough samples for FID computation")
    
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
    start_epoch = 1
    
    # Determine if this is the main process (rank 0) for saving
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    
    # Load from checkpoint if enabled and checkpoint exists
    if config.get('resume_from_checkpoint', False):
        checkpoint_path = config.get('checkpoint_to_load', None)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_main_process:
                print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Handle loading from different parallelization schemes
            g_state_dict = checkpoint['G']
            d_state_dict = checkpoint['D']
            
            # Remove 'module.' prefix if present (from DataParallel checkpoint)
            if list(g_state_dict.keys())[0].startswith('module.'):
                g_state_dict = {k.replace('module.', ''): v for k, v in g_state_dict.items()}
                d_state_dict = {k.replace('module.', ''): v for k, v in d_state_dict.items()}
            
            # Load into model
            if dist.is_initialized():
                G.module.load_state_dict(g_state_dict)
                D.module.load_state_dict(d_state_dict)
            elif isinstance(G, nn.DataParallel):
                G.module.load_state_dict(g_state_dict)
                D.module.load_state_dict(d_state_dict)
            else:
                G.load_state_dict(g_state_dict)
                D.load_state_dict(d_state_dict)
            
            opt_G.load_state_dict(checkpoint['opt_G'])
            opt_D.load_state_dict(checkpoint['opt_D'])
            start_epoch = checkpoint['epoch'] + 1
            best_fid = checkpoint.get('best_fid', float("inf"))
            
            if is_main_process:
                print(f"Resumed from epoch {checkpoint['epoch']}")
                print(f"Best FID so far: {best_fid:.2f}")
                print(f"Continuing training from epoch {start_epoch}\n")
        else:
            if is_main_process:
                print(f"\nWarning: resume_from_checkpoint=True but checkpoint not found at {checkpoint_path}")
                print("Starting training from scratch...\n")

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        G.train()
        D.train()
        running_D_loss = 0.0
        running_G_loss = 0.0
        running_real_acc = 0.0
        running_fake_acc = 0.0
        batches = 0

        # Only show progress bar on main process
        if is_main_process:
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
        else:
            train_bar = train_loader
            
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
            
            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
            # update progress bar (only on main process)
            if is_main_process and hasattr(train_bar, 'set_postfix'):
                train_bar.set_postfix({
                    'D_loss': f'{running_D_loss/batches:.3f}',
                    'G_loss': f'{running_G_loss/batches:.3f}'
                })

        # Synchronize all processes before validation
        if dist.is_initialized():
            dist.barrier()

        # Validation with limited FID samples
        if is_main_process:
            print("Running validation...")
        val_loss, val_racc, val_facc, fid_score = run_validation(
            G, D, val_loader, config['latent_dim'], device,
            max_fid_samples=config.get('max_fid_samples', 5000)
        )

        if is_main_process:
            print(f"Epoch [{epoch}/{config['num_epochs']}] "
                  f"Train D loss: {running_D_loss/batches:.4f} | "
                  f"Train G loss: {running_G_loss/batches:.4f} | "
                  f"Train D real acc: {running_real_acc/batches:.3f} | "
                  f"Train D fake acc: {running_fake_acc/batches:.3f} || "
                  f"Val D loss: {val_loss:.4f} | Val real acc: {val_racc:.3f} | "
                  f"Val fake acc: {val_facc:.3f} | FID: {fid_score:.2f}")

        # Save checkpoint every 3 epochs (only on main process)
        if epoch % 3 == 0 and is_main_process:
            print(f"Saving checkpoint at epoch {epoch}...")
            
            # Get state dict from DDP module if needed
            if dist.is_initialized():
                g_state = G.module.state_dict()
                d_state = D.module.state_dict()
            else:
                g_state = G.state_dict()
                d_state = D.state_dict()
            
            checkpoint_data = {
                "G": g_state,
                "D": d_state,
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch,
                "fid": fid_score,
                "train_d_loss": running_D_loss/batches,
                "train_g_loss": running_G_loss/batches,
                "val_d_loss": val_loss,
                "best_fid": best_fid
            }
            torch.save(checkpoint_data, 
                      os.path.join(config['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pth"))
        
        # Save sample every 5 epochs (only on main process)
        if (epoch % 5 == 0 or epoch == 1) and is_main_process:
            print(f"Saving sample images...")
            G.eval()
            with torch.no_grad():
                sample_z = gen_rand_noise(16, config['latent_dim'], device)
                fake_imgs = G(sample_z)
                fake_imgs = (fake_imgs + 1) / 2.0
                utils.save_image(fake_imgs, os.path.join(config['sample_dir'], f"fake_epoch_{epoch}.png"), nrow=4)
            G.train()

        # Synchronize before checking early stopping
        if dist.is_initialized():
            dist.barrier()

        # Early stopping based on FID (only main process saves best model)
        if fid_score < best_fid - 0.5:
            best_fid = fid_score
            epochs_no_improve = 0
            
            if is_main_process:
                print(f"New best FID! ({fid_score:.2f}) Saving...")
                
                # Get state dict from DDP module if needed
                if dist.is_initialized():
                    g_state = G.module.state_dict()
                    d_state = D.module.state_dict()
                else:
                    g_state = G.state_dict()
                    d_state = D.state_dict()
                
                torch.save({
                    "G": g_state,
                    "D": d_state,
                    "epoch": epoch,
                    "fid": fid_score
                }, os.path.join(config['model_dir'], f"gan_uncertainty_xray_best_{config['model_name_append']}.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                if is_main_process:
                    print(f"Early stopping at epoch {epoch} (best FID: {best_fid:.2f})")
                break
        
        # Clear cache after epoch
        torch.cuda.empty_cache()
    
    if is_main_process:
        print("Training complete!")

def test(G, config, device, test_loader):
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
    
    num_samples = min(2000, len(test_loader.dataset))
    samples_generated = 0
    
    # Collect some real images for side-by-side comparison
    real_samples = []
    
    with torch.no_grad():
        # Generate fake samples
        print(f"Generating {num_samples} fake samples...")
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
        for real, _ in test_loader:
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
        
        # Save comparison images
        utils.save_image(real_samples_scaled, 
                        os.path.join(config['test_dir'], "real_samples.png"), 
                        nrow=4, normalize=False)
        utils.save_image(fake_samples_scaled, 
                        os.path.join(config['test_dir'], "fake_samples.png"), 
                        nrow=4, normalize=False)
        
        # Side-by-side comparison
        comparison = torch.stack([real_samples_scaled, fake_samples_scaled], dim=1)
        comparison = comparison.view(-1, 1, config['img_height'], config['img_width'])
        utils.save_image(comparison, 
                        os.path.join(config['test_dir'], "real_vs_fake_comparison.png"), 
                        nrow=8, normalize=False)
    
    # Compute metrics
    print("\nComputing final metrics...")
    fid_score = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    
    print(f"\n{'='*50}")
    print(f"Test Results (NIH Chest X-ray Dataset):")
    print(f"{'='*50}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    print(f"\nSample images saved to: {config['test_dir']}")
    print(f"{'='*50}")

def setup_distributed():
    """Initialize distributed training for both torchrun and SLURM"""
    # Check if running with torchrun
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank
    
    # Check if running with SLURM
    elif 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ.get('SLURM_NTASKS', 1))
        
        # Only initialize distributed training if world_size > 1
        if world_size > 1:
            rank = int(os.environ['SLURM_PROCID'])
            local_rank = int(os.environ['SLURM_LOCALID'])
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            torch.cuda.set_device(local_rank)
            return local_rank
        else:
            # Single GPU SLURM job - no distributed training needed
            print("SLURM detected but only 1 task - running in single-GPU mode")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            # Don't set device - let CUDA_VISIBLE_DEVICES handle it
            return None  # Return None to signal single-GPU mode
    
    # No distributed training
    return None  # Return None for single-GPU/CPU mode

if __name__ == "__main__":
    os.environ['HF_HOME'] = '/blue/azare/lukesaleh/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/blue/azare/lukesaleh/.cache/huggingface/datasets'
    os.environ['TORCH_HOME'] = '/blue/azare/lukesaleh/.cache/torch'
    os.environ['TORCH_HUB'] = '/blue/azare/lukesaleh/.cache/torch/hub'
    
    # Setup distributed training if available
    local_rank = setup_distributed()
    
    # Set device based on local_rank
    if local_rank is not None:
        # Distributed mode - use specific device
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        # Single GPU mode - use default cuda device
        device = torch.device("cuda")
    else:
        # CPU mode
        device = torch.device("cpu")
    
    # Performance diagnostics
    print(f"\n{'='*70}")
    print(f"SYSTEM CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"{'='*70}\n")

    # Config for NIH dataset
    use_cnn = True
    
    if use_cnn:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Scale batch size with GPU count
        if num_gpus >= 4:
            base_batch_size = 256  # 1024 total for 4 GPUs
        elif num_gpus >= 2:
            base_batch_size = 128  # 256-512 total
        else:
            base_batch_size = 64   # Single GPU
        
        config = {
            'batch_size': base_batch_size * num_gpus,
            'latent_dim': 128,
            'num_epochs': 200,
            'lr_g': 2e-4,
            'lr_d': 5e-5,
            'beta1': 0.5,
            'beta2': 0.9,
            'lambda_gp': 10.0,
            'uncertainty_lambda': 1.0,
            'n_critic': 3,
            'patience': 20,
            'run_test': True,
            'use_cnn_generator': True,
            'img_height': 128,      
            'img_width': 128,       
            'img_channels': 1,
            'max_fid_samples': 5000,
            'resume_from_checkpoint': False,
            'checkpoint_to_load': None
        }
        
        print(f"\nMulti-GPU Setup:")
        print(f"  GPUs: {num_gpus}")
        print(f"  Total batch size: {config['batch_size']}")
        print(f"  Per-GPU batch size: {base_batch_size}")
    
    # Set model type
    if config['use_cnn_generator']:
        model_type = "cnn"
        config['model_name_append'] = "cnn"
    else:
        model_type = "mlp"
        config['model_name_append'] = "mlp"
    
    config['model_dir'] = os.path.join(os.getcwd(), "models", "models_uncertainty_nih", model_type)
    config['checkpoint_dir'] = os.path.join(os.getcwd(), "models", "models_uncertainty_nih", f"{model_type}_checkpoints")
    config['sample_dir'] = os.path.join(os.getcwd(), "results", "uncertainty_nih", model_type, "samples")
    config['test_dir'] = os.path.join(os.getcwd(), "results", "uncertainty_nih", model_type, "test")
    
    config['model_to_load'] = os.path.join(config['model_dir'], f"gan_uncertainty_xray_best_{model_type}.pth")

    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    os.makedirs(config['test_dir'], exist_ok=True)
    
    # Auto-detect latest checkpoint if resume is enabled
    if config['resume_from_checkpoint'] and config['checkpoint_to_load'] is None:
        checkpoint_files = [f for f in os.listdir(config['checkpoint_dir']) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            # Extract epoch numbers and find the latest
            epochs = [int(f.split('_')[-1].replace('.pth', '')) for f in checkpoint_files]
            latest_epoch = max(epochs)
            config['checkpoint_to_load'] = os.path.join(config['checkpoint_dir'], f"checkpoint_epoch_{latest_epoch}.pth")
            print(f"\nAuto-detected latest checkpoint: checkpoint_epoch_{latest_epoch}.pth")
        else:
            print("\nNo checkpoints found, will train from scratch")

    print(f"\n{'='*70}")
    print(f"Dataset: NIH Chest X-ray (Hugging Face)")
    print(f"Model Type: {model_type.upper()}")
    print(f"Image Size: {config['img_height']}x{config['img_width']}")
    print(f"Model Directory: {config['model_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Sample Directory: {config['sample_dir']}")
    print(f"Test Directory: {config['test_dir']}")
    if config['resume_from_checkpoint']:
        print(f"Resume from checkpoint: ENABLED")
        if config['checkpoint_to_load']:
            print(f"Checkpoint to load: {config['checkpoint_to_load']}")
    else:
        print(f"Resume from checkpoint: DISABLED (training from scratch)")
    print(f"{'='*70}\n")

    # Get data loaders from Hugging Face
    # Use max_samples for testing (e.g., max_samples=5000) or None for full dataset
    train_loader, val_loader, test_loader = get_data_loaders(
        config['batch_size'],
        img_height=config['img_height'],
        img_width=config['img_width'],
        val_split=0.15,
        use_augmentation=True,
        max_samples=None 
    )
    
    # Initialize models (on CPU first)
    if config['use_cnn_generator']:
        print(f"Using CNN-based generator for {config['img_height']}x{config['img_width']} X-ray images")
        G = Generator_CNN_Uncertainty(
            z_dim=config['latent_dim'],
            img_channels=config['img_channels'],
            img_height=config['img_height'],
            img_width=config['img_width']
        )
        D = Critic(
            img_channels=config['img_channels'],
            img_height=config['img_height'],
            img_width=config['img_width']
        )
    else:
        print("Using fully connected generator")
        G = Generator_Uncertainty(config['latent_dim'])
        D = Critic_Simple()
    
    # Print model info before parallelization
    print("Models initialized")
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    # Move to device FIRST, then wrap with DDP/DataParallel
    if dist.is_initialized():
        # DistributedDataParallel: move to device first, then wrap
        G = G.to(device)
        D = D.to(device)
        G = DDP(G, device_ids=[local_rank])
        D = DDP(D, device_ids=[local_rank])
        print(f"Using DistributedDataParallel on GPU {local_rank}")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # DataParallel: move to device first, then wrap
        G = G.to(device)
        D = D.to(device)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    else:
        # Single GPU or CPU: just move to device
        G = G.to(device)
        D = D.to(device)
        if torch.cuda.is_available():
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (no GPU available)")

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))
    opt_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
    
    if config['run_test']:
        test(G, config, device, test_loader)
    else:
        train(G, D, opt_G, opt_D, train_loader, val_loader, config, device)
