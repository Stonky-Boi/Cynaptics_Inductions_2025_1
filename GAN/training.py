import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm.auto import tqdm

# Define transformations
transform = transforms.Compose([
    transforms.Resize(128),  # Resize images to 128x128
    transforms.CenterCrop(128),  # Crop to 128x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Path to your dataset folder
dataset_path = '/kaggle/input/grapevine-leaves-image-dataset/Grapevine_Leaves_Image_Dataset'  # Replace with your folder path

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Generator class
class Generator(nn.Module):
    def __init__(self, z_dim=100, out_dim=3*128*128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, out_dim),
            nn.Tanh()  # Output between [-1, 1]
        )

    def forward(self, z):
        return self.gen(z).view(-1, 3, 128, 128)  # Output shape should be 3x128x128

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, in_dim=3*128*128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability between [0, 1]
        )

    def forward(self, x):
        return self.disc(x.view(x.size(0), -1))  # Flatten the image

# Initialize models
z_dim = 100  # Latent vector size
gen = Generator(z_dim=z_dim).cuda()
disc = Discriminator().cuda()

# Optimizers
lr = 0.0002
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
loss_func = nn.BCELoss()

# Function to generate noise
def generate_noise(batch_size, z_dim):
    return torch.randn(batch_size, z_dim).cuda()

# Function to show generated images
def show_images(tensor, num_images=16, size=(128, 128)):
    data = tensor.detach().cpu().view(-1, 3, *size)
    grid = make_grid(data[:num_images], nrow=4, padding=2).permute(1, 2, 0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# Training loop
epochs = 100
for epoch in range(epochs):
    gen_losses = []
    disc_losses = []

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as progress_bar:
        for real_images, _ in progress_bar:
            real_images = real_images.cuda()

            # Discriminator Training
            disc_opt.zero_grad()

            # Real images
            real_labels = torch.ones(real_images.size(0), 1).cuda()
            real_preds = disc(real_images)
            disc_real_loss = loss_func(real_preds, real_labels)

            # Fake images
            noise = generate_noise(real_images.size(0), z_dim)
            fake_images = gen(noise)
            fake_labels = torch.zeros(real_images.size(0), 1).cuda()
            fake_preds = disc(fake_images.detach())
            disc_fake_loss = loss_func(fake_preds, fake_labels)

            # Total discriminator loss
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            # Generator Training
            gen_opt.zero_grad()

            fake_preds = disc(fake_images)
            gen_loss = loss_func(fake_preds, real_labels)  # We want to fool the discriminator
            gen_loss.backward()
            gen_opt.step()

            # Track losses
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            # Update progress bar description
            progress_bar.set_postfix(
                gen_loss=f"{sum(gen_losses)/len(gen_losses):.4f}",
                disc_loss=f"{sum(disc_losses)/len(disc_losses):.4f}"
            )

        # Show generated images every epoch
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                noise = generate_noise(16, z_dim)
                generated_images = gen(noise)
                show_images(generated_images)

    print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {sum(gen_losses)/len(gen_losses):.4f}, Discriminator Loss: {sum(disc_losses)/len(disc_losses):.4f}")
