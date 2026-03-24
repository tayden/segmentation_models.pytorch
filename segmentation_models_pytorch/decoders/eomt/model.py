import warnings
from typing import Any, Optional, Union, Callable, Literal

import torch

from segmentation_models_pytorch.base import (
    SegmentationHead,
    ClassificationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders.timm_vit import TimmViTEncoder
from segmentation_models_pytorch.base.utils import is_torch_compiling
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from .decoder import EoMTDecoder


class EoMT(SegmentationModel):
    """EoMT (Encoder-only Mask Transformer) segmentation model.

    Adapts the EoMT architecture (CVPR 2025) as a transformer decoder with query
    cross-attention to ViT encoder features, with lightweight ConvTranspose upscaling.

    The decoder takes ViT features, applies transformer decoder layers where learnable
    queries cross-attend to spatial features, then produces per-query masks via
    dot-product with upscaled pixel features.

    Note:
        Since this model uses a Vision Transformer backbone, it typically requires a
        fixed input image size. You can set ``dynamic_img_size=True`` in the model
        initialization (if supported by the specific ``timm`` encoder).

    Args:
        encoder_name: Name of the ViT encoder. Must start with ``"tu-"``.
        encoder_depth: Number of stages used in encoder in range [1, 4].
        encoder_weights: One of **None** (random initialization), or not **None**
            (pretrained weights loaded).
        encoder_output_indices: Indices of encoder output features to use.
        decoder_readout: Strategy to utilize prefix tokens (e.g. cls_token).
            One of ``"cat"``, ``"add"``, or ``"ignore"``.
        decoder_hidden_dim: Hidden dimension for the transformer decoder and queries.
        decoder_num_queries: Number of learnable query tokens. Acts as the channel
            dimension of the decoder output.
        decoder_num_blocks: Number of transformer decoder layers.
        decoder_num_heads: Number of attention heads in transformer layers.
        decoder_num_upscale_blocks: Number of ScaleBlocks for upsampling pixel features.
            Each block doubles spatial resolution.
        in_channels: Number of input channels for the model.
        classes: Number of classes for output mask.
        activation: Activation function after final convolution.
        aux_params: Parameters for auxiliary classification head.
        kwargs: Arguments passed to the encoder.

    Returns:
        ``torch.nn.Module``: EoMT
    """

    _is_torch_scriptable = False
    _is_torch_compilable = True
    requires_divisible_input_shape = True

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "tu-vit_base_patch16_224.augreg_in21k",
        encoder_depth: int = 4,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_indices: Optional[list[int]] = None,
        decoder_readout: Literal["ignore", "add", "cat"] = "cat",
        decoder_hidden_dim: int = 256,
        decoder_num_queries: int = 100,
        decoder_num_blocks: int = 4,
        decoder_num_heads: int = 8,
        decoder_num_upscale_blocks: int = 2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_name.startswith("tu-"):
            encoder_name = encoder_name[3:]
        else:
            raise ValueError(
                f"Only Timm encoders are supported for EoMT. Encoder name must start with 'tu-', got {encoder_name}"
            )

        if decoder_readout not in ["ignore", "add", "cat"]:
            raise ValueError(
                f"Invalid decoder readout mode. Must be one of: 'ignore', 'add', 'cat'. Got: {decoder_readout}"
            )

        self.encoder = TimmViTEncoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=encoder_weights is not None,
            output_indices=encoder_output_indices,
            **kwargs,
        )

        if not self.encoder.has_prefix_tokens and decoder_readout != "ignore":
            warnings.warn(
                f"Encoder does not have prefix tokens (e.g. cls_token), but `decoder_readout` is set to '{decoder_readout}'. "
                f"It's recommended to set `decoder_readout='ignore'` when using an encoder without prefix tokens.",
                UserWarning,
            )

        self.decoder = EoMTDecoder(
            encoder_out_channels=self.encoder.out_channels,
            encoder_output_strides=self.encoder.output_strides,
            encoder_has_prefix_tokens=self.encoder.has_prefix_tokens,
            readout=decoder_readout,
            hidden_dim=decoder_hidden_dim,
            num_queries=decoder_num_queries,
            num_blocks=decoder_num_blocks,
            num_heads=decoder_num_heads,
            num_upscale_blocks=decoder_num_upscale_blocks,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_num_queries,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=4,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "eomt-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` through model's encoder, decoder and heads"""

        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features, prefix_tokens = self.encoder(x)
        decoder_output = self.decoder(features, prefix_tokens)
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
