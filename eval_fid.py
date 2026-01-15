import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from model import DDPM
from dpm_train import get_data, betas, alphas, alphas_cumprod, device

checkpoint_path = "diffusion_model_ema.pth"
num_batches =  5
batch_size = 32

# FID requires images to be
# 1. 3 channels (RGB)
# 2. uint8 type (0-255)
# 3. 299 x 299 resolution
def preprocess_for_fid(images):
    images = images.to("cpu")

    # 1. Denormalize from [-1, 1] to [0, 1] (if necessary)
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    # 2. Resize to 299 x 299
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

    # 3. Convert from 1-channel to 3-channel (RGB) -> (B, 1, H, W) to (B, 3, H, W)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    # 4. Scale to [0, 255] + convert to uint8
    images = (images * 255).to(torch.uint8)

    return images

@torch.no_grad()
def generate_batch(model, batch_size, T, device):
    """
    Generates a single batch of images from the model.
    """

    model.eval()

    # Start from pure noise
    x = torch.randn(batch_size, 1, 32, 32).to(device)
    for i in reversed(range(T)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        beta = betas[i]
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # Reverse equation
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    return x
    
def calculate_fid():
    print(f"Initializing FID metric on {device}")

    fid = FrechetInceptionDistance(feature=64).to("cpu")

    # 1. Load model
    model = DDPM(
                image_size=32,
                bottleneck_dim=4,
                in_channels=1,
                out_dim=1
                ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully.")

    # 2. Update with real images
    dataloader = get_data(train=False)
    print("Processing Real Images...")

    real_img_count = 0
    for i, (real_images, _) in enumerate(dataloader):
        if i > num_batches: break

        real_images = real_images.to(device)
        
        # Preprocess
        real_images = preprocess_for_fid(real_images)
        fid.update(real_images, real=True)
        real_img_count += real_images.shape[0]

    print(f"  -> Added {real_img_count} real images.")

    # 3. Update with FAKE (Generated) images
    # Generate the same number of batches as we pulled from real data
    print("Generating and Processing Fake Images...")
    for _ in tqdm(range(num_batches)):
        fake_images = generate_batch(model, batch_size, T=1000, device=device)
        
        # Preprocess
        fake_images = preprocess_for_fid(fake_images)
        
        # Update metric (real=False)
        fid.update(fake_images, real=False)

    # 4. Compute Score
    print("Computing FID score...")
    score = fid.compute()
    print(f"FID Score: {score.item():.4f}")

if __name__ == "__main__":
    calculate_fid()