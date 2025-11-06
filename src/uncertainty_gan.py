import os
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Config
# =========================
batch_size = 64
latent_dim = 100
num_epochs = 100
lr = 2e-4
beta1, beta2 = 0.5, 0.9
lambda_gp = 10.0
uncertainty_lambda = 0.5   
n_critic = 5               
patience = 10              
model_dir = os.getcwd() + "/models/models_uncertainty"
sample_dir = os.getcwd() + "/results/uncertainty/samples"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

# Turn this on to run test/inference instead of training
RUN_TEST = False
MODEL_TO_LOAD = os.path.join(model_dir, "gan_uncertainty_fmnist_best.pth")

# =========================
# Data Transforms and Loaders
# =========================
print("Loading data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    # WGAN often uses [-1,1] input; FMNIST is (0,1), so scale:
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
print(f"Train samples: {train_len}, Val samples: {val_len}")

# =========================
# Models
# =========================

class Generator_Uncertainty(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()  
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)

class Critic(nn.Module):
    # WGAN critic: no sigmoid
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 7x7
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)

G = Generator_Uncertainty(latent_dim).to(device)
D = Critic().to(device)
print("Models initialized")

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

# Gradient penalty for WGAN-GP during training
def gradient_penalty(critic, real, fake):
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
def gen_rand_noise(n, z_dim):
    return torch.randn(n, z_dim, device=device)

def critic_accuracy(real_scores, fake_scores):
    # real>fake threshold
    real_correct = (real_scores > 0).float().mean().item()
    fake_correct = (fake_scores < 0).float().mean().item()
    return real_correct, fake_correct

# =========================
# Training and Validation
# =========================

def run_validation():
    D.eval()
    G.eval()
    val_critic_loss = 0.0
    val_real_acc = 0.0
    val_fake_acc = 0.0
    count = 0
    
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for real, _ in val_bar:
            real = real.to(device)
            z = gen_rand_noise(real.size(0), latent_dim)
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

    D.train()
    G.train()
    return (val_critic_loss / count,
            val_real_acc / count,
            val_fake_acc / count)


def train():
    print("\nStarting training...")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()
        running_D_loss = 0.0
        running_G_loss = 0.0
        running_real_acc = 0.0
        running_fake_acc = 0.0
        batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for i, (real, _) in enumerate(train_bar):
            real = real.to(device)
            bsz = real.size(0) # get current batch size

            # Update critic n_critic times
            for _ in range(n_critic):
                z = gen_rand_noise(bsz, latent_dim)
                fake = G(z).detach()
                real_scores = D(real)
                fake_scores = D(fake)

                gp = gradient_penalty(D, real, fake)
                loss_D = (fake_scores.mean() - real_scores.mean()) + lambda_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            
            z = gen_rand_noise(bsz, latent_dim)
            fake = G(z)
            fake_scores_for_G = D(fake)

            # batch means
            with torch.no_grad():
                # get real scores for mean calculation
                real_scores_for_mean = D(real)
            mu_fake = fake_scores_for_G.mean().detach()
            mu_real = real_scores_for_mean.mean().detach()

            # uncertainty = how far above avg fake you are, capped by gap
            gap = (mu_real - mu_fake).clamp(min=1e-6)
            uncertainty = (fake_scores_for_G - mu_fake).clamp(min=0.0, max=gap)

            loss_G_adv = -fake_scores_for_G.mean()
            loss_G_unc = -uncertainty_lambda * uncertainty.mean()

            loss_G = loss_G_adv + loss_G_unc

            opt_G.zero_grad()
            loss_G.backward()
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
        val_loss, val_racc, val_facc = run_validation()

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train D loss: {running_D_loss/batches:.4f} | "
              f"Train G loss: {running_G_loss/batches:.4f} | "
              f"Train D real acc: {running_real_acc/batches:.3f} | "
              f"Train D fake acc: {running_fake_acc/batches:.3f} || "
              f"Val D loss: {val_loss:.4f} | Val real acc: {val_racc:.3f} | Val fake acc: {val_facc:.3f}")

        # Save sample every 5 epochs
        if epoch % 10 == 0:
            print(f"Saving sample images...")
            G.eval()
            with torch.no_grad():
                sample_z = gen_rand_noise(16, latent_dim)
                fake_imgs = G(sample_z)
                fake_imgs = (fake_imgs + 1) / 2.0  # back to [0,1]
                utils.save_image(fake_imgs, os.path.join(sample_dir, f"fake_epoch_{epoch}.png"), nrow=4)
            G.train()

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"New best model! Saving...")
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "epoch": epoch
            }, os.path.join(model_dir, "gan_fmnist_best.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break
    
    print("Training complete!")

def test():
    print("Loading model for testing...")
    # Load and generate images
    checkpoint = torch.load(MODEL_TO_LOAD, map_location=device)
    G.load_state_dict(checkpoint["G"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    print("Generating test images...")
    G.eval()
    with torch.no_grad():
        z = gen_rand_noise(16, latent_dim)
        fake_imgs = G(z)
        fake_imgs = (fake_imgs + 1) / 2.0
        utils.save_image(fake_imgs, os.path.join(sample_dir, "fake_test.png"), nrow=4)
    print("Test samples saved to:", os.path.join(sample_dir, "fake_test.png"))

if __name__ == "__main__":
    if RUN_TEST:
        test()
    else:
        train()
