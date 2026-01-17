import torch

@torch.no_grad()
def ddim_sample(model,
                n_samples,
                image_size,
                in_channels,
                alphas_cumprod,
                y,
                device,
                timesteps=1000,
                ddim_steps=50,
                eta=0.0,
                seed=None
                ):
    
    # Generator
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # Setup time sequence
    c = timesteps // ddim_steps
    time_seq = list(reversed(range(0, timesteps, c)))

    # Start noise
    x = torch.randn(n_samples, in_channels, image_size, image_size, generator=generator).to(device)

    # Sampling loop
    for i, t_cur in enumerate(time_seq):
        # Next timestep (t_prev)
        t_prev = -1 if i == len(time_seq) - 1 else time_seq[i+1]

        # Predict noise
        t_tensor = torch.full((n_samples,), t_cur, device=device, dtype=torch.long)
        noise_pred = model(x, t_tensor, y)

        # Calculate alphas
        alpha_bar_cur = alphas_cumprod[t_cur]
        alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)

        # DDIM update
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_cur) * noise_pred) / torch.sqrt(alpha_bar_cur)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_prev))
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * noise_pred

        noise = torch.randn(x.shape, generator=generator, device=device) if eta > 0 else 0.
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        yield x, pred_x0

@torch.no_grad()
def ddpm_sample(model,
                n_samples,
                image_size,
                in_channels,
                betas,
                alphas,
                alphas_cumprod,
                y,
                device,
                timesteps=1000,
                seed=None
                ):
    
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # Start noise
    x = torch.randn(n_samples, in_channels, image_size, image_size, generator=generator).to(device)

    # Sampling loop
    for i in reversed(range(timesteps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        predicted_noise = model(x, t, y)

        beta = betas[i]
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]

        if i > 0:
            noise = torch.randn(x.shape, generator=generator).to(device)
        else:
            noise = torch.zeros_like(x)

        # Langevin dynamics
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        yield x, x