import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

from diffusion_core.model import DiffusionModel
from diffusion_core.sampling import ddim_sample

device = "mps" if torch.backends.mps.is_available() else "cpu"

def pipeline(image_tensor, prompt, device="mps"):
    print("Loading ControlNet and SD1.5 models (this might take a moment)...")

    img_array = (image_tensor.clamp(-1, 1) + 1) / 2
    img_array = (img_array * 255).type(torch.uint8).cpu().squeeze().numpy()
    init_image = Image.fromarray(img_array, mode="L").convert("RGB")
    init_image = init_image.resize((512, 512), Image.NEAREST)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16
    )

    sd = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    sd.scheduler = UniPCMultistepScheduler.from_config(sd.scheduler.config)
    sd = sd.to(device)
    sd.enable_attention_slicing()

    # Generate output
    print(f"Generating stylized image for prompt: '{prompt}'")

    output = sd(
        prompt,
        image=init_image,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0
    ).images[0]

    return output

def main(args):
    print("--- Starting Base Model Generation ---")

    if not os.path.exists(args.model_path):
        raise RuntimeError("Model not found! Please check the path")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint["config"]
    model_cfg = config["model"]

    model = DiffusionModel(**model_cfg).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    alphas_cumprod = torch.linspace(0.9999, 0.02, 1000) # Dummy. Pull this from schedule.py based on config

    if args.digit is not None:
        y = torch.tensor([args.digit]).long().to(device)
        print(f"Generating digit: {args.digit}")
    else:
        y = torch.randint(0, 10, (1,)).long().to(device)
        print(f"Generating random digit: {y.item()}")

    sampler = ddim_sample(
        model,
        n_samples=1,
        image_size=model_cfg["image_size"],
        in_channels=model_cfg["in_channels"],
        alphas_cumprod=alphas_cumprod,
        y=y,
        device=device,
        seed=args.seed
    )

    final_image = None
    for _, pred_x0 in sampler:
        final_img = pred_x0

    final_img_save = (final_img.clamp(-1, 1) + 1) / 2
    plt.imsave("base_digit.png", final_img_save.cpu().squeeze(), cmap="gray")
    print("Base digit generated and saved as 'base_digit.png'.")

    # Run Stable Diffusion wrapper
    sd_result = pipeline(final_img, args.prompt, device)
    sd_result.save("stylized_digit.png")
    print("Stylized digit generated and saved as 'stylized_digit.png'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/diffusion_model_final.pth")
    parser.add_argument("--digit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, required=True)

    args = parser.parse_args()
    main(args)

