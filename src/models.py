import torch
import torch.nn as nn
import math

# Fully Connected Generator for FMNIST (28x28)
class Generator_Uncertainty(nn.Module):
    """Original fully connected generator for FashionMNIST"""
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


# CNN-based Generator with flexible dimensions
class Generator_CNN_Uncertainty(nn.Module):
    """
    CNN-based generator with customizable output dimensions
    Works with any image size 
    
    """
    def __init__(self, z_dim=100, img_channels=1, img_height=28, img_width=28, 
                 base_channels=64, init_size=4):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.init_size = init_size
        
        # Calculate number of upsampling layers needed
        num_upsample_h = max(0, math.ceil(math.log2(img_height / init_size)))
        num_upsample_w = max(0, math.ceil(math.log2(img_width / init_size)))
        num_upsample = max(num_upsample_h, num_upsample_w)
        
        # Calculate intermediate size after all conv2dtranspose layers
        self.intermediate_h = init_size * (2 ** num_upsample)
        self.intermediate_w = init_size * (2 ** num_upsample)
        
        # Initial projection from latent to feature maps
        self.init_channels = base_channels * (2 ** num_upsample)
        self.fc = nn.Sequential(
            nn.Linear(z_dim, self.init_channels * init_size * init_size),
            nn.BatchNorm1d(self.init_channels * init_size * init_size),
            nn.ReLU(True)
        )
        
        # Build upsampling layers
        layers = []
        in_channels = self.init_channels
        
        for i in range(num_upsample):
            out_channels = in_channels // 2 if i < num_upsample - 1 else base_channels
            
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
            in_channels = out_channels
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # Use adaptive convolution or cropping/padding
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Upsample
        x = self.conv_blocks(x)
        
        # Apply final convolution
        x = self.final_conv(x)
        
        # Adjust to exact target size if needed
        if x.shape[2] != self.img_height or x.shape[3] != self.img_width:
            # Use adaptive pooling or interpolation to get exact size
            x = nn.functional.interpolate(x, size=(self.img_height, self.img_width), 
                                         mode='bilinear', align_corners=False)
        
        return x


# Fully Connected Generator with flexible dimensions
class Generator_FC_Uncertainty(nn.Module):
    """
    Fully connected generator that works with any image dimensions.
    """
    def __init__(self, z_dim=100, img_channels=1, img_height=28, img_width=28):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.output_size = img_channels * img_height * img_width
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_size),
            nn.Tanh()  
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, self.img_channels, self.img_height, self.img_width)


# Critic/Discriminator (Works with any image size)
class Critic(nn.Module):
    """
    WGAN critic with no sigmoid activation, adapts to any input image size.
    """
    def __init__(self, img_channels=1, img_height=28, img_width=28, base_channels=64):
        super().__init__()
        
        layers = []
        in_channels = img_channels
        current_h = img_height
        current_w = img_width
        
        # Downsample until we reach a small spatial size
        while current_h > 4 or current_w > 4:
            out_channels = base_channels if len(layers) == 0 else min(in_channels * 2, 512)
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            in_channels = out_channels
            current_h = (current_h + 1) // 2  # Ceiling division for odd sizes
            current_w = (current_w + 1) // 2
        
        # Use adaptive pooling to handle any remaining size
        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate final features
        final_features = in_channels * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_features, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x.view(-1)


# Simple Critic for 28x28 images (Original)
class Critic_Simple(nn.Module):
    """Simple critic specifically for FashionMNIST"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)