import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .blocks import AttentionBlock, AdaGNResidualBlock

# Positional Emncoding for the timestep
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()

        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        # Implementing 1 / (10000)^scale
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)

        return embeddings
    
class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()

        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, dim):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        # Nearest Neighbor
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
class DiffusionModel(nn.Module):
    def __init__(self,
                 image_size=64,
                 in_channels=3,
                 model_channels=64,
                 bottleneck_dim=4,
                 max_channels=512,
                 out_dim=3,
                 time_emb_dim=256,
                 num_classes=None
                 ):
        
        super(DiffusionModel, self).__init__()
        
        # Class conditioning setup
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # Calculating required depth
        # e. g. 64 // 4 -> 16 -> log2(16) = 4 steps
        if image_size % bottleneck_dim != 0:
            raise ValueError(f"Image size {image_size} must be divisible by bottleneck {bottleneck_dim}")
        
        ratio = image_size // bottleneck_dim
        num_resolutions = int(math.log2(ratio))

        # Auto-generate channel multipliers
        channel_mults = []
        current_mult = 1

        for i in range(num_resolutions):
            channel_mults.append(current_mult)

            # Double the channels until we hit the max
            if(model_channels * current_mult * 2) <= max_channels:
                current_mult *= 2

        # --- UNet construction ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.conv0 = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = model_channels
        curr_channels = [channels] # Track skip connections

        # --- Dynamic DOWN PATH ---
        for idx, mult in enumerate(channel_mults):
            out_channels = model_channels * mult

            # We add a Downsample layer at the END of every block except the last one
            # to ensure we reach the exact bottleneck dimension at the end
            is_last = idx == (len(channel_mults) - 1)

            self.downs.append(nn.ModuleList([
                AdaGNResidualBlock(channels, out_channels, time_emb_dim),
                AdaGNResidualBlock(out_channels, out_channels, time_emb_dim),
                Downsample(out_channels) if not is_last else nn.Identity()
            ]))

            channels = out_channels
            curr_channels.append(channels)

        # --- BOTTLENECK ---
        self.mid1 = AdaGNResidualBlock(channels, channels, time_emb_dim)
        self.attn = AttentionBlock(channels)
        self.mid2 = AdaGNResidualBlock(channels, channels, time_emb_dim)

        # --- Dynamic UP PATH ---
        for idx, mult in enumerate(reversed(channel_mults)):
            out_channels = model_channels * mult

            # Corresponding skip connections
            skip_channels = curr_channels.pop()

            self.ups.append(nn.ModuleList([
                Upsample(channels) if idx > 0 else nn.Identity(),
                AdaGNResidualBlock(channels + skip_channels, out_channels, time_emb_dim),
                AdaGNResidualBlock(out_channels, out_channels, time_emb_dim)
            ]))

            channels = out_channels

        self.output = nn.Conv2d(channels, out_dim, 1)

    def forward(self, x, t, y=None):
        t = self.time_mlp(t)

        # Inject Class Information
        if y is not None and self.num_classes is not None:
            t += self.label_emb(y)

        x = self.conv0(x)
        skips = [x]

        # --- DOWN PATH ---
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skips.append(x)
            x = downsample(x)

        # --- BOTTLENECK ---
        x = self.mid1(x, t)
        x = self.attn(x)
        x = self.mid2(x, t)

        # --- UP PATH ---
        for upsample, block1, block2 in self.ups:
            if not isinstance(upsample, nn.Identity):
                x = upsample(x)

            skip = skips.pop()
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)

        return self.output(x)