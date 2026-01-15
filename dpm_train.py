import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import DDPM
from utils import LiveLossPlot
from ema import EMA

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Config
batch_size = 128
epochs = 20
lr = 1e-4
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas.cpu(), axis=0).to(device)

def get_data(train=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def forward_diffusion(x_0, t):
    """
    Takes an image and a timestep t.
    Returns the noisy image x_t and the specific noise noise_epsilon added.
    """

    noise = torch.randn_like(x_0)

    # Get the alpha value for this specific timestep t
    # Reshape to [batch_size, 1, 1, 1] for broadcasting
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    # The Equation: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    return x_t, noise

if __name__ == "__main__":

    dataloader = get_data()
    model = DDPM(
                image_size=32,
                bottleneck_dim=4,
                in_channels=1,
                out_dim=1,
                num_classes=10
                ).to(device)
    
    # Init EMA
    ema = EMA(model)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    diffusion_loss = nn.MSELoss()

    plotter = LiveLossPlot()
    step_count = 0

    for epoch in range(epochs):
        for step, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Sample random timestamps for every image in the batch
            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            x_t, noise = forward_diffusion(images, t)

            noise_pred = model(x_t, t, labels)

            loss = diffusion_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            ema.update_model_average(model)

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                plotter.update(loss.item())
                step_count += 1

    plt.ioff()
    plt.show()

    torch.save(model.state_dict(), f"diffusion_model.pth")
    ema.save_checkpoint("diffusion_model_ema.pth")
    print(f"--> Finished. Models saved.")