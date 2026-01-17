import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self,
                 channels,
                 num_groups=8
                 ):
        
        super(AttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1) # Compute Query, Key, Value all at once
        self.proj = nn.Conv2d(channels, channels, 1) # Projection layer for the output

    def forward(self, x):
        B, C, H, W = x.shape

        # 1 Normalize and compute Q, K, V
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # 2 Reshape (Flash attention expects: B, Heads, SeqLen, Dim)
        q = q.reshape(B, 1, C, -1).permute(0, 1, 3, 2)
        k = k.reshape(B, 1, C, -1).permute(0, 1, 3, 2)
        v = v.reshape(B, 1, C, -1).permute(0, 1, 3, 2)

        # 3 Compute Flash attention (= self attention optimized)
        h = F.scaled_dot_product_attention(q, k, v)

        # 4 Reshape back
        h = h.squeeze(1).permute(0, 2, 1).reshape(B, C, H, W)

        # 5 Projection + Residual
        return x + h
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 time_emb_dim, 
                 num_groups=8
                 ):
        
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Linear projection for time embeddings
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)

        # Add time embedding (broadcast to spatial dims)
        h = h + self.time_proj(t)[:, :, None, None]

        h = self.conv2(h)
        return h + self.shortcut(x)
    
class AdaGNResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim,
                 num_groups=8,
                 dropout=0.1
                 ):
        
        super(AdaGNResidualBlock, self).__init__()

        # First convolution block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Adaptive Group Normalization (AdaGN) Projection
        # Projects time/class embedding to predict Scale (gamma) and Shift (beta)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        # Second convolution block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        # Calculate scale and shift from embedding
        emb_out = self.emb_proj(t_emb)

        # Reshape to (B, C, 1, 1) for broadcasting
        # scale, shift shape: [Batch, Out_Channels, 1, 1]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        # AdaGN
        h = self.norm2(h)
        h = h * (1 + scale) + shift

        h = self.conv2(F.silu(h))
        h = self.dropout(h)

        return h + self.shortcut(x)