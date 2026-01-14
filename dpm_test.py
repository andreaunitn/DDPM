import torch

import matplotlib.pyplot as plt

from model import DDPM

device = "mps" if torch.backends.mps.is_available() else "cpu"

T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas.cpu(), axis=0).to(device)

@torch.no_grad()
def sample_image(model_path):
    model = DDPM(
                image_size=32,
                bottleneck_dim=4,
                in_channels=1,
                out_dim=1
                ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Start from pure noise
    img = torch.randn(1, 1, 28, 28).to(device)

    for i in reversed(range(T)):
        t = torch.tensor([i], device=device).long()
        
        predicted_noise = model(img, t)

        # Coefficients
        beta = betas[i]
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]

        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        # The Reverse Equation (Langevin Dynamics)
        # Subtract the predicted noise to "clean" the image
        img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    img = (img + 1) / 2
    img = img.clamp(0, 1)

    img = img.cpu().squeeze().numpy()
    plt.imsave("digit.png", img, cmap="gray")

if __name__ == "__main__":
    sample_image("diffusion_model_ema.pth")