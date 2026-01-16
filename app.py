import torch
from PIL import Image

import imageio
import gradio as gr

from model import DDPM

device = "cpu"
model_path = "diffusion_model_ema.pth"

model = DDPM(
            image_size=32,
            bottleneck_dim=4,
            in_channels=1,
            out_dim=1,
            num_classes=10
            ).to(device)

if torch.backends.mps.is_available():
    checkpoint = torch.load(model_path, map_location="mps")
else:
    checkpoint = torch.load(model_path, map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)

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
def generate_digit(digit):
    """
    Input: int (0-9)
    Output: PIL Image (256x256)
    """
    
    # Handle the input from the UI
    y = torch.tensor([digit]).long().to(device)
    img = torch.randn(1, 1, 32, 32).to(device)

    intermediate_steps = []

    # Fast sampling for web (optional: you could skip steps to make it faster)
    for i in reversed(range(T)):
        t = torch.tensor([i], device=device).long()
        predicted_noise = model(img, t, y)

        beta = betas[i]
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]

        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        # Capture frames for visual feedback
        if i % 100 == 0 and i != 0:
            step_img = process_image(img)
            intermediate_steps.append((step_img, f"Step {i}"))

            yield None, intermediate_steps

        if i == 0:
            final_image = process_image(img)
            intermediate_steps.append((final_image, "Final Result"))

            yield final_image, intermediate_steps

# --- INTERFACE ---
with gr.Blocks(title="Diffusion Process") as demo:
    gr.Markdown("# 🎨 Conditional Diffusion Demo")
    gr.Markdown("Pick a number and watch the Diffusion Model builts it from pure noise.")
    
    with gr.Row():
        with gr.Column(scale=1):
            digit_input = gr.Slider(0, 9, step=1, label="Which number?", value=5)
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            final_output = gr.Image(label="Final Result", type="pil")
            gallery = gr.Gallery(label="Denoising Process", columns=4, height="auto")

    run_btn.click(fn=generate_digit, inputs=digit_input, outputs=[final_output, gallery])

if __name__ == "__main__":
    demo.launch()