import torch
import math

def get_cosine_schedule(T, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)

    # Cosine function
    f_t = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2

    # Normalize so that it starts from 0
    alphas_cumprod = f_t / f_t[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.5) # Clip to prevent singularities

def get_linear_schedule(T):
    return torch.linspace(1e-4, 0.02, T)