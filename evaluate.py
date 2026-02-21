import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import argparse
import os
import glob
import math
import re
import gc

from diffusion_core.model import DiffusionModel
from diffusion_core.schedule import get_cosine_schedule
from diffusion_core.sampling import ddim_sample

device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_real_images_loader(batch_size):
    """
    Loads real MNIST images.
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def numerical_sort(value):
    numbers = re.findall(r"\d+", value)
    return int(numbers[-1]) if numbers else -1

def evaluate(args):

    checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "*.pth"))
    checkpoints.sort(key=numerical_sort)

    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints.")

    base_fid = FrechetInceptionDistance(feature=2048, normalize=True)
    base_fid = base_fid.to("cpu")

    is_metric = InceptionScore(feature="logits_unbiased", splits=10)
    is_metric = is_metric.to("cpu")

    # Pre-calculate real statistics
    print("\nPre-calculating statistics for REAL images...")

    real_batch_size = 100
    real_loader = get_real_images_loader(batch_size=real_batch_size)
    total_fed = 0

    for batch, _ in real_loader:
        if total_fed >= args.n_samples:
            break
        
        batch = batch.to("cpu")
        batch_rgb = batch.repeat(1, 3, 1, 1)
        batch_rgb = (batch_rgb.clamp(-1, 1) + 1) / 2
        base_fid.update(batch_rgb, real=True)

        total_fed += batch.shape[0]
        if total_fed % 2000 == 0:
            print(f" -> Processed {total_fed} real images")

    print(f"Done. Stats ready for {total_fed} images.")

    # Config loading
    first_ckpt_path = checkpoints[0]
    print(f"Loading architecture config from {os.path.basename(first_ckpt_path)}")
    first_ckpt = torch.load(first_ckpt_path, map_location=device)

    if "config" in first_ckpt:
        conf = first_ckpt["config"]
        model_cfg = conf["model"]
        sample_cfg = conf["sample"]
    else:
        model_cfg = {"image_size": 32, "bottleneck_dim": 4, "in_channels": 1,
                     "out_dim": 1, "num_classes": 10, "model_channels": 64,
                     "max_channels": 512, "time_emb_dim": 256
                    }
        sample_cfg = {"T": 1000, "ddim_steps": 50}

    model = DiffusionModel(
                image_size=model_cfg["image_size"],
                bottleneck_dim=model_cfg["bottleneck_dim"],
                in_channels=model_cfg["in_channels"],
                out_dim=model_cfg["out_dim"],
                num_classes=model_cfg["num_classes"],
                model_channels=model_cfg["model_channels"],
                max_channels=model_cfg["max_channels"],
                time_emb_dim=model_cfg["time_emb_dim"],
                ).to(device)
    
    T = sample_cfg["T"]
    betas = get_cosine_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas.cpu(), axis=0).to(device)

    results = []
    for ckpt_path in checkpoints:
        filename = os.path.basename(ckpt_path)
        print(f"--- Processing {filename} ---")

        checkpoint = torch.load(ckpt_path, map_location=device)
        if "ema_model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        current_fid = base_fid.clone()
        is_metric.reset()

        # Generation loop (batched)
        # Generating 10k images divided in chunks of 50

        gen_batch_size = 50
        total_generated = 0
        num_batches = math.ceil(args.n_samples / gen_batch_size)
        
        print(f"Generating {args.n_samples} fake images...")
        for i in range(num_batches):
            current_batch_size = min(gen_batch_size, args.n_samples - total_generated)

            y = torch.randint(0, 10, (current_batch_size, )).long().to(device)
            sampler = ddim_sample(
                model, 
                current_batch_size, 
                model_cfg["image_size"], 
                model_cfg["in_channels"], 
                alphas_cumprod, y, device, 
                ddim_steps=sample_cfg["ddim_steps"], 
                guidance_scale=sample_cfg["guidance_scale"], 
                null_class_idx=sample_cfg["null_class_idx"]
                )

            fake_batch = None
            for x, _ in sampler:
                fake_batch = x

            fake_batch_rgb = fake_batch.repeat(1, 3, 1, 1).to("cpu")
            fake_norm = (fake_batch_rgb.clamp(-1, 1) + 1) / 2
            current_fid.update(fake_norm, real=False)

            fake_uint8 = (fake_norm * 255).to(torch.uint8)
            is_metric.update(fake_uint8)

            total_generated += current_batch_size
            del fake_batch, fake_batch_rgb, fake_norm, fake_uint8

            if total_generated % 2000 == 0:
                print(f" -> Generated {total_generated}/{args.n_samples}")
                gc.collect()

        print("Computing metrics...")

        fid_score = current_fid.compute().item()
        is_score, is_std = is_metric.compute()
        is_score = is_score.item()
        print(f"FID: {fid_score:.4f} | IS: {is_score:.4f}")
        results.append((filename, fid_score, is_score))

        del current_fid
        gc.collect()

    # Summary
    print("\n\n" + "="*80)
    print(f"{'Checkpoint':<35} | {'FID (Lower=Better)':<20} | {'IS (Higher=Better)':<20}")
    print("-" * 80)

    best_fid = float("inf")
    best_ckpt_fid = ""

    for name, f_score, i_score in results:
        print(f"{name:<35} | {f_score:<20.4f} | {i_score:<20.4f}")
        if f_score < best_fid:
            best_fid = f_score
            best_ckpt_fid = name

    print("-" * 80)
    print(f"Best model by FID: {best_ckpt_fid} ({best_fid:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--n_samples", type=int, default=10000)
    args = parser.parse_args()

    evaluate(args)
