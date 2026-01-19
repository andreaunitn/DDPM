import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import argparse
import os

from diffusion_core.model import DiffusionModel
from diffusion_core.schedule import get_cosine_schedule
from diffusion_core.utils import LiveLossPlot, save_training_checkpoint
from diffusion_core.ema import EMA

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Config
batch_size = 128
epochs = 50
lr = 2e-4
T = 1000
betas = get_cosine_schedule(T).to(device)
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

def main(args):

    dataloader = get_data()
    model = DiffusionModel(
                image_size=32,
                bottleneck_dim=4,
                in_channels=1,
                out_dim=1,
                num_classes=10
                ).to(device)
    
    # Init EMA
    ema = EMA(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    diffusion_loss = nn.MSELoss()

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(device)

    plotter = LiveLossPlot()
    start_epoch = 0
    step_count = 0

    # Resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            ema.step = checkpoint["ema_step"]

            start_epoch = checkpoint["epoch"] + 1
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            raise RuntimeError("Model not found!")
        
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        for step, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            # Sample random timestamps for every image in the batch
            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            x_t, noise = forward_diffusion(images, t)

            # Autocast for mixed precision
            with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
                noise_pred = model(x_t, t, labels)
                loss = diffusion_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            ema.update_model_average(model)
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

            if step % 50 == 0:
                plotter.update(loss.item())
                step_count += 1

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_training_checkpoint(f"checkpoints/diffusion_model_{epoch}.pth", epoch, model, optimizer, scaler, ema)
            print(f"Saved checkpoint for epoch {epoch}")

    plt.ioff()
    plt.show()

    save_training_checkpoint(f"checkpoints/diffusion_model_final.pth", epochs, model, optimizer, scaler, ema)
    ema.save_checkpoint("checkpoints/diffusion_model_ema_final.pth")
    print(f"--> Finished. Models saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    main(args)