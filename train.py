import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

from diffusion_core.model import DiffusionModel
from diffusion_core.schedule import get_cosine_schedule
from diffusion_core.utils import save_training_checkpoint
from diffusion_core.ema import EMA
from diffusion_core.sampling import ddim_sample

device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_data(batch_size, train=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=True,
                      num_workers=4,
                      pin_memory=False,
                      persistent_workers=True
                      )

def forward_diffusion(x_0, t, alphas_cumprod):
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

@torch.no_grad()
def sample_validation_images(model, alphas_cumprod, device, writer, global_step):
    model.eval()

    num_classes = 10
    y = torch.arange(num_classes).long().to(device)

    sampler = ddim_sample(
        model=model,
        n_samples=num_classes,
        image_size=32,
        in_channels=1,
        alphas_cumprod=alphas_cumprod,
        y=y,
        device=device,
        timesteps=1000,
        ddim_steps=50
    )

    x_gen = None
    for x, _ in sampler:
        x_gen = x

    x_gen = (x_gen.clamp(-1, 1) + 1) / 2

    grid = make_grid(x_gen, nrow=5)
    writer.add_image("Validation/Generates_Images", grid, global_step)

    model.train()

def main(args):

    # Config
    batch_size = args.batch_size
    grad_accumulation_steps = args.grad_acc_steps
    epochs = args.epochs
    lr = args.lr
    T = args.timesteps
    betas = get_cosine_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas.cpu(), axis=0).to(device)

    writer = SummaryWriter(log_dir="runs/diffusion_mnist")

    dataloader = get_data(batch_size)
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
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(dataloader) // grad_accumulation_steps,
        epochs=epochs
    )

    diffusion_loss = nn.MSELoss()

    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(device)

    start_epoch = 0
    global_step = 0

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
            global_step = checkpoint["global_step"]
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            raise RuntimeError("Model not found!")
        
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            # Sample random timestamps for every image in the batch
            t = torch.randint(0, T, (images.shape[0],), device=device).long()
            x_t, noise = forward_diffusion(images, t, alphas_cumprod)

            # Autocast for mixed precision
            with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
                noise_pred = model(x_t, t, labels)
                loss = diffusion_loss(noise_pred, noise)
                loss = loss / grad_accumulation_steps # Normalizing

            scaler.scale(loss).backward()

            if (step + 1) % grad_accumulation_steps == 0:

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()
                ema.update_model_average(model)

            global_step += 1
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

            if step % 50 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_training_checkpoint(f"checkpoints/diffusion_model_{epoch}.pth", epoch,  global_step, model, optimizer, scaler, ema)
            print(f"Saved checkpoint for epoch {epoch}")

            print("Generating validation samples...")
            validation_model = ema.ema_model
            sample_validation_images(
                validation_model,
                alphas_cumprod,
                device,
                writer,
                global_step
            )

    save_training_checkpoint(f"checkpoints/diffusion_model_final.pth", epochs, global_step, model, optimizer, scaler, ema)

    writer.close()
    print(f"--> Finished. Models saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Number of accumulation steps before updating the parameters")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    main(args)