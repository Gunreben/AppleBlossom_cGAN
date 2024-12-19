import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

#########################################
# Dataset Setup
#########################################

class BBCHDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Each subdirectory is a BBCH stage label
        classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            cdir = os.path.join(root_dir, c)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    fpath = os.path.join(cdir, fname)
                    self.samples.append((fpath, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#########################################
# Hyperparameters
#########################################

batch_size = 32
image_size = 64   # Adjust based on your dataset resolution
nz = 500         # Latent vector size
num_classes = 10
lr = 0.0002
beta1 = 0.5
epochs = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# Data Loading
#########################################

# Update the path to your dataset root directory
dataset_path = "/home/gunreben/Documents/cGAN/appleBlossomDataset/Apple_Blossom_Images/train"

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = BBCHDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#########################################
# Conditional GAN Architecture
#########################################

class Generator(nn.Module):
    def __init__(self, nz, ngf=64, num_classes=10):
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
    def __init__(self, ndf=64, num_classes=10):
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
# Instantiate Models and Optimizers
#########################################

netG = Generator(nz, ngf=64, num_classes=num_classes).to(device)
netD = Discriminator(ndf=64, num_classes=num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#########################################
# Training Loop
#########################################

fixed_noise = torch.randn(batch_size, nz, device=device)
fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)

for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train Discriminator
        optimizerD.zero_grad()
        b_size = real_images.size(0)
        real_target = torch.full((b_size, 1), 1.0, device=device)
        fake_target = torch.full((b_size, 1), 0.0, device=device)

        output = netD(real_images, labels)
        errD_real = criterion(output, real_target)

        noise = torch.randn(b_size, nz, device=device)
        fake_images = netG(noise, labels)
        output = netD(fake_images.detach(), labels)
        errD_fake = criterion(output, fake_target)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train Generator
        optimizerG.zero_grad()
        output = netD(fake_images, labels)
        errG = criterion(output, real_target)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

    # Save some generated samples every epoch
    with torch.no_grad():
        fake_samples = netG(fixed_noise, fixed_labels)
        save_image(fake_samples, f'output_epoch_{epoch}.png', normalize=True)

#########################################
# After training you can save your models
#########################################

torch.save(netG.state_dict(), "cgan_generator.pth")
torch.save(netD.state_dict(), "cgan_discriminator.pth")
