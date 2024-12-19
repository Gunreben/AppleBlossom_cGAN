import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

#########################################
# Define Dataset and Models (same as before)
#########################################

# Define the dataset, generator, and discriminator classes here
# (reuse the same implementations from your original training script)

class Generator(nn.Module):
    def __init__(self, nz, ngf=64, num_classes=5):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Linear(nz + num_classes, ngf*8*4*4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_onehot = torch.zeros(labels.size(0), self.label_emb.num_embeddings, device=labels.device)
        label_onehot.scatter_(1, labels.unsqueeze(1), 1)
        x = torch.cat([noise, label_onehot], 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=64, num_classes=5):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(ndf*4*8*8 + num_classes, 1)

    def forward(self, img, labels):
        batch_size = img.size(0)
        features = self.conv(img)
        features = features.view(batch_size, -1)
        label_onehot = torch.zeros(batch_size, self.label_emb.num_embeddings, device=labels.device)
        label_onehot.scatter_(1, labels.unsqueeze(1), 1)
        out = torch.cat([features, label_onehot], 1)
        out = self.fc(out)
        return out

#########################################
# Load Saved Models and Optimizers
#########################################

nz = 500  # Latent vector size
num_classes = 10  # Adjust to your dataset
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
netG = Generator(nz, ngf=64, num_classes=num_classes).to(device)
netD = Discriminator(ndf=64, num_classes=num_classes).to(device)

# Initialize optimizers
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# Load saved states
netG.load_state_dict(torch.load("cgan_generator.pth", map_location=device))
netD.load_state_dict(torch.load("cgan_discriminator.pth", map_location=device))

optimizerG.load_state_dict(torch.load("optimizerG.pth", map_location=device))
optimizerD.load_state_dict(torch.load("optimizerD.pth", map_location=device))

#########################################
# Resume Training
#########################################

# Load your dataset and dataloader as before
# Example: dataset_path = "/path/to/dataset"
# dataset = BBCHDataset(dataset_path, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs_to_train = 200  # Add more epochs
for epoch in range(epochs_to_train):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Continue training logic (same as original training loop)
        # Train Discriminator
        optimizerD.zero_grad()
        b_size = real_images.size(0)
        real_target = torch.full((b_size, 1), 1.0, device=device)
        fake_target = torch.full((b_size, 1), 0.0, device=device)

        output = netD(real_images, labels)
        errD_real = nn.BCEWithLogitsLoss()(output, real_target)

        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise, labels)
        output = netD(fake_images.detach(), labels)
        errD_fake = nn.BCEWithLogitsLoss()(output, fake_target)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train Generator
        optimizerG.zero_grad()
        output = netD(fake_images, labels)
        errG = nn.BCEWithLogitsLoss()(output, real_target)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs_to_train}] Step [{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

    # Save models and optimizers after each epoch
    torch.save(netG.state_dict(), "cgan_generator.pth")
    torch.save(netD.state_dict(), "cgan_discriminator.pth")
    torch.save(optimizerG.state_dict(), "optimizerG.pth")
    torch.save(optimizerD.state_dict(), "optimizerD.pth")
