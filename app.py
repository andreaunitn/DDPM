import torch
from PIL import Image

import gradio as gr
import os

from diffusion_core.model import DiffusionModel
from diffusion_core.schedule import get_cosine_schedule
from diffusion_core.sampling import ddpm_sample, ddim_sample

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_path = "checkpoints/diffusion_model_ema.pth"

# Load model
model = DiffusionModel(
            image_size=32,
            bottleneck_dim=4,
            in_channels=1,
            out_dim=1,
            num_classes=10
            ).to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
else:
    raise RuntimeError("Model not found!")

# Setup schedule
T = 1000
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
        sampler = ddim_sample(
            model, n_samples=1, image_size=32, in_channels=1,
            alphas_cumprod=alphas_cumprod, y=y, device=device,
            timesteps=T, ddim_steps=50, seed=seed
        )
        
        update_frequency = 5

    else:
        sampler = ddpm_sample(
            model, n_samples=1, image_size=32, in_channels=1,
            betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod,
            y=y, device=device, timesteps=T, seed=seed
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