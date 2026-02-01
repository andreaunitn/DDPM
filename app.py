import torch
from PIL import Image

import gradio as gr
import argparse
import os

from diffusion_core.model import DiffusionModel
from diffusion_core.schedule import get_cosine_schedule
from diffusion_core.sampling import ddpm_sample, ddim_sample

device = "mps" if torch.backends.mps.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/diffusion_model_final.pth")
args = parser.parse_args() 

model_path = args.model_path
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")

checkpoint = torch.load(model_path, map_location=device)
if "config" in checkpoint:
    conf = checkpoint["config"]
    model_cfg = conf["model"]
    sample_cfg = conf["sample"]
else:
    model_cfg = {"image_size": 32, "bottleneck_dim": 4, "in_channels": 1,
                    "out_dim": 1, "num_classes": 10, "model_channels": 64,
                    "max_channels": 512, "time_emb_dim": 256
                }
    sample_cfg = {"T": 1000, "ddim_steps": 50}

# Load model
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

# Load weights
if "ema_model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["ema_model_state_dict"])
    print("Loaded EMA weights.")
else:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded standard weights.")

model.eval()

# Setup schedule
T = sample_cfg["T"]
betas = get_cosine_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas.cpu(), axis=0).to(device)

def process_image(img_tensor):
    """
    Helper to convert raw Tensor (1, 1, 32, 32) -> Resized PIL Image (256, 256)
    """

    # Normalize to [-1, 1]
    img = (img_tensor.clamp(-1, 1) + 1) / 2

    # Convert to Numpy uint8 [0, 255]
    img = (img * 255).type(torch.uint8).cpu().squeeze().numpy()

    # Tranform to a PIL image
    pil_img = Image.fromarray(img, "L") # L = grayscale

    # Resize
    pil_img = pil_img.resize((256, 256), resample=Image.NEAREST)
    return pil_img

@torch.no_grad()
def generate_digit(digit, use_ddim, seed):
    """
    Input: int (0-9)
    Output: PIL Image (256 x 256)
    """
    
    if seed is not None:
        seed = int(seed)

    y = torch.tensor([digit]).long().to(device)
    intermediate_steps = []

    if use_ddim:
        sampler = ddim_sample(model,
                              n_samples=1,
                              image_size=model_cfg["image_size"],
                              in_channels=model_cfg["in_channels"],
                              alphas_cumprod=alphas_cumprod,
                              y=y,
                              device=device,
                              ddim_steps=sample_cfg["ddim_steps"],
                              seed=seed
                              )
        
        update_frequency = 5

    else:
        sampler = ddpm_sample(model,
                              n_samples=1,
                              image_size=model_cfg["image_size"],
                              in_channels=model_cfg["in_channels"],
                              betas=betas,
                              alphas=alphas,
                              alphas_cumprod=alphas_cumprod,
                              y=y,
                              device=device,
                              seed=seed
                              )

        update_frequency = 100

    # Generation loop
    for i, (x, pred_x0) in enumerate(sampler):
        if i % update_frequency == 0:
            current_img = process_image(x)
            intermediate_steps.append((current_img, f"Step {i}"))
            
            yield None, intermediate_steps

    final_clean = process_image(pred_x0)
    intermediate_steps.append((final_clean, "Final Result"))
    yield final_clean, intermediate_steps

# --- INTERFACE ---
with gr.Blocks(title="Diffusion Process") as demo:
    gr.Markdown("# 🎨 Conditional Diffusion Demo")
    gr.Markdown("Pick a number and watch the Diffusion Model builts it from pure noise.")
    
    with gr.Row():
        with gr.Column(scale=1):
            digit_input = gr.Slider(0, 9, step=1, label="Which number?", value=5)
            seed_input = gr.Number(label="Seed (leave blank for random)", value=None, precision=0)
            use_ddim_chk = gr.Checkbox(label="Use fast sampling (DDIM)", value=True)
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            final_output = gr.Image(label="Final Result", type="pil")
            gallery = gr.Gallery(label="Denoising Process", columns=4, height="auto")

    run_btn.click(fn=generate_digit, inputs=[digit_input, use_ddim_chk, seed_input], outputs=[final_output, gallery])

if __name__ == "__main__":
    demo.launch()