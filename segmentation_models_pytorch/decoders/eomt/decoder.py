from typing import Optional, Sequence, Literal

import torch
import torch.nn as nn

from segmentation_models_pytorch.decoders.dpt.decoder import (
    ReadoutConcatBlock,
    ReadoutAddBlock,
    ReadoutIgnoreBlock,
)


class ScaleBlock(nn.Module):
    """Lightweight upscaling block that doubles spatial resolution.

    Uses ConvTranspose2d for upsampling followed by depthwise convolution,
    batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.depthwise_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class EoMTTransformerDecoderLayer(nn.Module):
    """Standard transformer decoder layer with pre-norm.

    Consists of self-attention, cross-attention to encoder memory,
    and a feed-forward network, all with residual connections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(queries)
        queries = queries + self.self_attn(normed, normed, normed)[0]

        normed = self.norm2(queries)
        queries = queries + self.cross_attn(normed, memory, memory)[0]

        queries = queries + self.ffn(self.norm3(queries))

        return queries


def _get_2d_sinusoidal_positional_encoding(
    h: int, w: int, embed_dim: int
) -> torch.Tensor:
    """Generate 2D sinusoidal positional encoding."""
    half_dim = embed_dim // 2
    quarter_dim = half_dim // 2

    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    omega = torch.arange(quarter_dim, dtype=torch.float32) / quarter_dim
    omega = 1.0 / (10000.0**omega)

    pos_h = grid_h.reshape(-1, 1) * omega.unsqueeze(0)
    pos_w = grid_w.reshape(-1, 1) * omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(pos_h), torch.cos(pos_h), torch.sin(pos_w), torch.cos(pos_w)],
        dim=1,
    )
    return pe  # (H*W, embed_dim)


class EoMTDecoder(nn.Module):
    """EoMT (Encoder-only Mask Transformer) decoder.

    Takes ViT features, applies transformer decoder layers where learnable queries
    cross-attend to spatial features, then produces per-query masks via dot-product
    with upscaled pixel features.
    """

    def __init__(
        self,
        encoder_out_channels: Sequence[int],
        encoder_output_strides: Sequence[int],
        encoder_has_prefix_tokens: bool,
        readout: Literal["cat", "add", "ignore"] = "cat",
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_blocks: int = 4,
        num_heads: int = 8,
        num_upscale_blocks: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        encoder_dim = encoder_out_channels[-1]

        # Readout block for prefix tokens
        if readout == "cat":
            self.readout_block = ReadoutConcatBlock(
                encoder_dim, encoder_has_prefix_tokens
            )
        elif readout == "add":
            self.readout_block = ReadoutAddBlock()
        elif readout == "ignore":
            self.readout_block = ReadoutIgnoreBlock()
        else:
            raise ValueError(
                f"Invalid readout mode: {readout}, should be one of: 'cat', 'add', 'ignore'"
            )

        # Project encoder features to hidden_dim
        self.feature_proj = nn.Conv2d(encoder_dim, hidden_dim, kernel_size=1)

        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder layers
        self.transformer_layers = nn.ModuleList(
            [
                EoMTTransformerDecoderLayer(hidden_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Scale blocks for upsampling pixel features
        self.scale_blocks = nn.ModuleList(
            [ScaleBlock(hidden_dim, hidden_dim) for _ in range(num_upscale_blocks)]
        )

        # Mask projection head
        self.mask_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

    def forward(
        self,
        features: list[torch.Tensor],
        prefix_tokens: list[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        # Take deepest encoder feature
        feat = features[-1]  # (B, C_enc, H_p, W_p)
        prefix = prefix_tokens[-1]

        # Handle prefix tokens via readout
        feat = self.readout_block(feat, prefix)  # (B, C_enc, H_p, W_p)

        # Project to hidden_dim
        feat = self.feature_proj(feat)  # (B, hidden_dim, H_p, W_p)

        # Flatten to sequence + add positional encoding
        B, C, H, W = feat.shape
        memory = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)

        # 2D sinusoidal positional encoding
        spatial_pos = _get_2d_sinusoidal_positional_encoding(H, W, self.hidden_dim)
        spatial_pos = spatial_pos.to(memory.device, dtype=memory.dtype)
        memory = memory + spatial_pos.unsqueeze(0)

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer decoder layers
        for layer in self.transformer_layers:
            queries = layer(queries, memory)  # (B, num_queries, hidden_dim)

        # Pixel branch - upsample spatial features
        pixel_features = feat  # (B, hidden_dim, H_p, W_p)
        for scale_block in self.scale_blocks:
            pixel_features = scale_block(pixel_features)

        # Mask prediction via dot-product
        query_proj = self.mask_proj(queries)  # (B, num_queries, hidden_dim)
        masks = torch.einsum("bqc,bchw->bqhw", query_proj, pixel_features)

        return masks
