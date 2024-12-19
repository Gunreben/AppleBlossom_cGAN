import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

#########################################
# Generator Model Definition (same as training)
#########################################
class Generator(torch.nn.Module):
    def __init__(self, nz, ngf=64, num_classes=5):  # Update num_classes to 5 to match your model
        super(Generator, self).__init__()
        self.label_emb = torch.nn.Embedding(num_classes, num_classes)
        self.fc = torch.nn.Linear(nz + num_classes, ngf*8*4*4)

        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf*4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf*2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        label_onehot = torch.zeros(labels.size(0), self.label_emb.num_embeddings, device=labels.device)
        label_onehot.scatter_(1, labels.unsqueeze(1), 1)

        x = torch.cat([noise, label_onehot], 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.main(x)
        return x

#########################################
# Parameters and Setup
#########################################
nz = 500  # Latent vector size
num_classes = 10  # Number of BBCH stages (adjust to match your model)
image_size = 256
output_path = "generated_image.png"

# Load the trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, ngf=64, num_classes=num_classes).to(device)
netG.load_state_dict(torch.load("cgan_generator.pth", map_location=device))
netG.eval()

#########################################
# Generate a Single Image
#########################################
def generate_image(label, output_path, image_size=64):
    """
    Generates a single image for a given label and resizes it to the specified image size.

    Args:
        label (int): Class label for the BBCH stage (e.g., 0 to num_classes-1).
        output_path (str): File path to save the generated image.
        image_size (int): Desired output image size (default: 64).
    """
    # Validate label
    if label < 0 or label >= num_classes:
        raise ValueError(f"Label must be between 0 and {num_classes - 1}, got {label}.")

    # Generate noise and label tensors
    noise = torch.randn(1, nz, device=device)
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)

    # Generate image
    with torch.no_grad():
        generated_image = netG(noise, label_tensor).cpu()

    # Resize the image to the specified size
    transform = transforms.Resize((image_size, image_size))
    resized_image = transform(generated_image)

    # Save the image
    save_image(resized_image, output_path, normalize=True)
    print(f"Generated image saved to {output_path}")

#########################################
# Example Usage
#########################################
if __name__ == "__main__":
    # Change the label to the desired BBCH stage (e.g., 0 to num_classes-1)
    label = 8  # Example: Generate an image for BBCH stage 0
    generate_image(label, output_path, image_size=image_size)
