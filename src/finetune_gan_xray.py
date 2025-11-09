import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Import Hugging Face datasets
from datasets import load_dataset

# Import torchmetrics for FID and precision/recall
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Import models
from models import Generator_Uncertainty, Generator_CNN_Uncertainty, Critic, Critic_Simple

# Import distributed training utilities
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Define collate_fn at module level for Windows compatibility
def collate_fn(batch):
    """Custom collate function to handle data from HuggingFace dataset"""
    images = []
    labels = []
    
    for item in batch:
        # Convert from list back to tensor
        img = torch.tensor(item['pixel_values'], dtype=torch.float32)
        images.append(img)
        labels.append(item['label'])
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_data_loaders(batch_size, img_height=128, img_width=128, val_split=0.15, max_samples=None):
    """
    Load NIH Chest X-ray dataset from Hugging Face WITHOUT augmentation
    Only basic transforms: resize, convert to tensor, normalize
    """
    print("="*70)
    print("Loading NIH Chest X-ray dataset from Hugging Face (NO AUGMENTATION)...")
    print("This may take a few minutes on first run (dataset will be cached)")
    print("="*70)

    try:
        # Load dataset with trust_remote_code
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            name="image-classification",
            split="train",
            cache_dir=os.environ.get('HF_DATASETS_CACHE'),
            trust_remote_code=True
        )
        
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print(f"Using subset of {max_samples:,} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Split dataset BEFORE any transforms
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
    
    print(f"\n{'='*70}")
    print(f"NIH Chest X-ray Dataset Statistics (Fine-tuning):")
    print(f"{'='*70}")
    print(f"Training Set:   {len(train_dataset):,} images (NO AUGMENTATION)")
    print(f"Validation Set: {len(val_dataset):,} images")
    print(f"Test Set:       {len(test_dataset):,} images")
    print(f"Total:          {total_size:,} images")
    print(f"\nImage Size:     {img_height}x{img_width}")
    print(f"Transforms applied on-the-fly during loading")
    print(f"{'='*70}\n")
    
    # Define basic transforms - NO AUGMENTATION
    basic_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    
    # Custom collate function that applies transforms on-the-fly
    def collate_fn_with_transform(batch):
        images = []
        labels = []
        
        for item in batch:
            img = item['image']
            if img.mode != 'L':
                img = img.convert('L')
            img = basic_transform(img)
            images.append(img)
            labels.append(0)  # Dummy label
        
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    # Optimize workers for multi-GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    cpu_count = os.cpu_count() if os.cpu_count() is not None else 8
    num_workers = min(4 * num_gpus, cpu_count - 2)
    num_workers = max(num_workers, 2)
    
    # Check if distributed training is initialized
    is_distributed = dist.is_initialized()
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_transform,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_transform,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_transform,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


# Gradient penalty for WGAN-GP during training
def gradient_penalty(critic, real, fake, device):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1), device=device).expand_as(real)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


def gen_rand_noise(n, z_dim, device):
    return torch.randn(n, z_dim, device=device)


def critic_accuracy(real_scores, fake_scores):
    real_correct = (real_scores > 0).float().mean()
    fake_correct = (fake_scores < 0).float().mean()
    return (real_correct + fake_correct) / 2


def add_label_to_image(img_tensor, label, font_size=20):
    img = transforms.ToPILImage()(img_tensor.cpu())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), label, fill=255, font=font)
    return transforms.ToTensor()(img)


def run_validation(G, D, val_loader, latent_dim, device, max_fid_samples=5000):
    """Run validation metrics"""
    G.eval()
    D.eval()
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    with torch.no_grad():
        real_count = 0
        fake_count = 0
        
        # Process real images
        for real_imgs, _ in val_loader:
            if real_count >= max_fid_samples:
                break
            real_imgs = real_imgs.to(device)
            batch_size = min(real_imgs.size(0), max_fid_samples - real_count)
            real_imgs = real_imgs[:batch_size]
            
            # Convert grayscale to RGB for FID
            real_rgb = real_imgs.repeat(1, 3, 1, 1)
            real_rgb = ((real_rgb + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fid.update(real_rgb, real=True)
            real_count += batch_size
        
        # Generate fake images
        while fake_count < max_fid_samples:
            batch_size = min(64, max_fid_samples - fake_count)
            noise = gen_rand_noise(batch_size, latent_dim, device)
            fake_imgs = G(noise)
            
            fake_rgb = fake_imgs.repeat(1, 3, 1, 1)
            fake_rgb = ((fake_rgb + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake_rgb, real=False)
            fake_count += batch_size
    
    fid_score = fid.compute().item()
    G.train()
    D.train()
    
    return fid_score


def finetune(G, D, opt_G, opt_D, train_loader, val_loader, config, device):
    """Fine-tune the pre-trained generator on unaugmented data"""
    
    best_fid = float('inf')
    patience_counter = 0
    
    is_distributed = dist.is_initialized()
    is_main = not is_distributed or dist.get_rank() == 0
    
    if is_main:
        print(f"\n{'='*70}")
        print(f"Starting Fine-tuning (NO AUGMENTATION)")
        print(f"{'='*70}")
        print(f"Initial learning rates - G: {config['lr_g']:.2e}, D: {config['lr_d']:.2e}")
        print(f"Lambda GP: {config['lambda_gp']}")
        print(f"N-Critic: {config['n_critic']}")
        print(f"Patience: {config['patience']}")
        print(f"{'='*70}\n")
    
    for epoch in range(1, config['num_epochs'] + 1):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        G.train()
        D.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_gp = 0.0
        epoch_d_acc = 0.0
        num_batches = 0
        
        # Training loop
        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch_idx, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Train Critic
            for _ in range(config['n_critic']):
                noise = gen_rand_noise(batch_size, config['latent_dim'], device)
                fake_imgs = G(noise).detach()
                
                real_scores = D(real_imgs)
                fake_scores = D(fake_imgs)
                
                gp = gradient_penalty(D, real_imgs, fake_imgs, device)
                d_loss = -(real_scores.mean() - fake_scores.mean()) + config['lambda_gp'] * gp
                
                opt_D.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
                opt_D.step()
                
                epoch_d_loss += d_loss.item()
                epoch_gp += gp.item()
                
                with torch.no_grad():
                    d_acc = critic_accuracy(real_scores, fake_scores)
                    epoch_d_acc += d_acc.item()
            
            # Train Generator
            noise = gen_rand_noise(batch_size, config['latent_dim'], device)
            fake_imgs = G(noise)
            fake_scores = D(fake_imgs)
            g_loss = -fake_scores.mean()
            
            opt_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            opt_G.step()
            
            epoch_g_loss += g_loss.item()
            num_batches += 1
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            if is_main and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}',
                    'GP': f'{gp.item():.4f}',
                    'D_acc': f'{d_acc.item():.4f}'
                })
        
        # Average losses
        avg_d_loss = epoch_d_loss / (num_batches * config['n_critic'])
        avg_g_loss = epoch_g_loss / num_batches
        avg_gp = epoch_gp / (num_batches * config['n_critic'])
        avg_d_acc = epoch_d_acc / (num_batches * config['n_critic'])
        
        # Synchronize all processes before validation
        if is_distributed:
            dist.barrier()
        
        # Run validation
        if is_main:
            print("\nRunning validation...")
        
        val_d_loss, val_real_acc, val_fake_acc, fid_score = run_validation_detailed(
            G, D, val_loader, config['latent_dim'], device,
            max_fid_samples=config.get('max_fid_samples', 5000)
        )
        
        if is_main:
            print(f"\nEpoch [{epoch}/{config['num_epochs']}] "
                  f"Train D loss: {avg_d_loss:.4f} | "
                  f"Train G loss: {avg_g_loss:.4f} | "
                  f"Train D acc: {avg_d_acc:.3f} | "
                  f"Train GP: {avg_gp:.4f} || "
                  f"Val D loss: {val_d_loss:.4f} | "
                  f"Val real acc: {val_real_acc:.3f} | "
                  f"Val fake acc: {val_fake_acc:.3f} | "
                  f"FID: {fid_score:.2f}")
        
        # Save sample images every 5 epochs or first epoch
        if (epoch % 5 == 0 or epoch == 1) and is_main:
            print(f"Saving sample images...")
            G.eval()
            with torch.no_grad():
                sample_z = gen_rand_noise(64, config['latent_dim'], device)
                fake_imgs = G(sample_z)
                fake_imgs_scaled = (fake_imgs + 1) / 2.0
                utils.save_image(
                    fake_imgs_scaled,
                    os.path.join(config['sample_dir'], f"finetuned_epoch_{epoch:03d}_fid_{fid_score:.2f}.png"),
                    nrow=8,
                    normalize=False
                )
            G.train()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 and is_main:
            print(f"Saving checkpoint at epoch {epoch}...")
            model_to_save = G.module if isinstance(G, DDP) else G
            checkpoint_data = {
                'epoch': epoch,
                'G': model_to_save.state_dict(),
                'D': D.module.state_dict() if isinstance(D, DDP) else D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'fid': fid_score,
                'train_d_loss': avg_d_loss,
                'train_g_loss': avg_g_loss,
                'val_d_loss': val_d_loss,
                'best_fid': best_fid,
                'config': config
            }
            torch.save(checkpoint_data, 
                      os.path.join(config['checkpoint_dir'], f"finetuned_checkpoint_epoch_{epoch:03d}.pth"))
        
        # Synchronize before checking early stopping
        if is_distributed:
            dist.barrier()
        
        # Early stopping based on FID
        if fid_score < best_fid - 0.5:
            best_fid = fid_score
            patience_counter = 0
            
            if is_main:
                print(f"New best FID! ({fid_score:.2f}) Saving best model...")
                
                model_to_save = G.module if isinstance(G, DDP) else G
                torch.save({
                    'epoch': epoch,
                    'G': model_to_save.state_dict(),
                    'D': D.module.state_dict() if isinstance(D, DDP) else D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict(),
                    'fid': fid_score,
                    'config': config
                }, os.path.join(config['model_dir'], 
                               f"gan_uncertainty_xray_finetuned_best_{config['model_name_append']}.pth"))
        else:
            patience_counter += 1
            if is_main:
                print(f"No improvement. Patience: {patience_counter}/{config['patience']}")
            
            if patience_counter >= config['patience']:
                if is_main:
                    print(f"\n{'='*70}")
                    print(f"Early stopping triggered after {epoch} epochs")
                    print(f"Best FID: {best_fid:.2f}")
                    print(f"{'='*70}")
                break
        
        # Clear cache after epoch
        torch.cuda.empty_cache()
    
    if is_main:
        print(f"\n{'='*70}")
        print(f"Fine-tuning Complete!")
        print(f"Best FID: {best_fid:.2f}")
        print(f"{'='*70}\n")
    
    return best_fid


def run_validation_detailed(G, D, val_loader, latent_dim, device, max_fid_samples=5000):
    """
    Run validation with detailed metrics including discriminator accuracy
    Similar to the validation in uncertainty_gan_xray.py
    """
    G.eval()
    D.eval()
    
    val_critic_loss = 0.0
    val_real_acc = 0.0
    val_fake_acc = 0.0
    count = 0
    
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Track samples processed for FID
    fid_samples_processed = 0
    compute_fid = True
    
    with torch.no_grad():
        for real, _ in val_loader:
            real = real.to(device)
            batch_size = real.size(0)
            z = gen_rand_noise(batch_size, latent_dim, device)
            fake = G(z)

            real_scores = D(real)
            fake_scores = D(fake)

            # WGAN critic loss 
            loss_D = (fake_scores.mean() - real_scores.mean()).item()
            val_critic_loss += loss_D

            # Accuracy
            r_acc = (real_scores > 0).float().mean().item()
            f_acc = (fake_scores < 0).float().mean().item()
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

    # Compute FID score
    if fid_samples_processed > 0:
        fid_score = fid.compute().item()
    else:
        fid_score = float('inf')
    
    G.train()
    D.train()
    
    return (val_critic_loss / count,
            val_real_acc / count,
            val_fake_acc / count,
            fid_score)


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
            return None
    
    # No distributed training
    return None


if __name__ == "__main__":
    os.environ['HF_HOME'] = '/blue/azare/lukesaleh/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/blue/azare/lukesaleh/.cache/huggingface/datasets'
    os.environ['TORCH_HOME'] = '/blue/azare/lukesaleh/.cache/torch'
    os.environ['TORCH_HUB'] = '/blue/azare/lukesaleh/.cache/torch/hub'
    
    # Setup distributed training if available
    local_rank = setup_distributed()
    
    # Set device based on local_rank
    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Performance diagnostics
    print(f"\n{'='*70}")
    print(f"SYSTEM CONFIGURATION (Fine-tuning)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"{'='*70}\n")

    # Configuration
    use_cnn = True
    
    if use_cnn:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Smaller batch size for fine-tuning (more stable)
        if num_gpus >= 4:
            base_batch_size = 128
        elif num_gpus >= 2:
            base_batch_size = 64
        else:
            base_batch_size = 32
        
        config = {
            'batch_size': base_batch_size * num_gpus,
            'latent_dim': 128,
            'num_epochs': 100,  # Fewer epochs for fine-tuning
            'lr_g': 1e-4,  # Lower learning rate
            'lr_d': 2e-5,  # Lower learning rate
            'beta1': 0.5,
            'beta2': 0.9,
            'lambda_gp': 10.0,
            'n_critic': 3,
            'patience': 15,
            'use_cnn_generator': True,
            'img_height': 128,
            'img_width': 128,
            'img_channels': 1,
            'max_fid_samples': 5000,
            'pretrained_model': '/blue/azare/lukesaleh/Uncertainty-Evolved-Multi-Gen-GANs/models/models_uncertainty_nih/cnn/gan_uncertainty_xray_best_cnn.pth'
        }
        
        print(f"\nFine-tuning Setup:")
        print(f"  GPUs: {num_gpus}")
        print(f"  Total batch size: {config['batch_size']}")
        print(f"  Per-GPU batch size: {base_batch_size}")
    
    # Set model type
    model_type = "cnn" if config['use_cnn_generator'] else "mlp"
    config['model_name_append'] = model_type
    
    # Set paths
    config['model_dir'] = os.path.join(os.getcwd(), "models", "models_uncertainty_nih_finetuned", model_type)
    config['checkpoint_dir'] = os.path.join(os.getcwd(), "models", "models_uncertainty_nih_finetuned", f"{model_type}_checkpoints")
    config['sample_dir'] = os.path.join(os.getcwd(), "results", "uncertainty_nih_finetuned", model_type, "samples")
    
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Fine-tuning Configuration")
    print(f"{'='*70}")
    print(f"Dataset: NIH Chest X-ray (NO AUGMENTATION)")
    print(f"Model Type: {model_type.upper()}")
    print(f"Pre-trained model: {config['pretrained_model']}")
    print(f"Image Size: {config['img_height']}x{config['img_width']}")
    print(f"Fine-tuned Model Directory: {config['model_dir']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"Sample Directory: {config['sample_dir']}")
    print(f"{'='*70}\n")
    
    # Load data (NO AUGMENTATION)
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=base_batch_size,
        img_height=config['img_height'],
        img_width=config['img_width'],
        val_split=0.15
    )
    
    # Initialize models
    print("Initializing models...")
    if config['use_cnn_generator']:
        G = Generator_CNN_Uncertainty(
            z_dim=config['latent_dim'],
            img_channels=config['img_channels'],
            img_height=config['img_height'],
            img_width=config['img_width']
        )
    else:
        G = Generator_Uncertainty(z_dim=config['latent_dim'])
    
    # Load pre-trained generator weights
    print(f"Loading pre-trained generator from: {config['pretrained_model']}")
    checkpoint = torch.load(config['pretrained_model'], map_location='cpu')
    G.load_state_dict(checkpoint['G'])
    print(f"Loaded pre-trained model from epoch {checkpoint['epoch']}")
    if 'fid' in checkpoint:
        print(f"Pre-trained FID: {checkpoint['fid']:.2f}")
    
    # Create new discriminator (fresh start for fine-tuning)
    D = Critic(
        img_channels=config['img_channels'],
        img_height=config['img_height'],
        img_width=config['img_width']
    )
    print("Initialized new discriminator for fine-tuning")
    
    # Move to device
    G = G.to(device)
    D = D.to(device)
    
    # Wrap with DDP if distributed
    is_distributed = dist.is_initialized()
    if is_distributed:
        G = DDP(G, device_ids=[local_rank])
        D = DDP(D, device_ids=[local_rank])
        print(f"Models wrapped with DistributedDataParallel (Rank {dist.get_rank()})")
    
    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(config['beta1'], config['beta2']))
    opt_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))
    
    print(f"\nStarting fine-tuning on unaugmented data...")
    print(f"This will create a stronger generator by training on clean data\n")
    
    # Fine-tune
    best_fid = finetune(G, D, opt_G, opt_D, train_loader, val_loader, config, device)
    
    print(f"\n{'='*70}")
    print(f"Fine-tuning complete!")
    print(f"Best FID achieved: {best_fid:.2f}")
    print(f"Fine-tuned model saved to: {config['model_dir']}")
    print(f"{'='*70}\n")
    
    # Cleanup distributed
    if is_distributed:
        dist.destroy_process_group()