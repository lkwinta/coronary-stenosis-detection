from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block used inside the decoder.

    This block applies two consecutive convolution + batch normalization + ReLU
    operations.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, out_channels, H, W].
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Single decoder stage: upsample, concatenate skip connection, refine.

    Parameters
    ----------
    in_channels : int
        Number of channels in the decoder input.
    skip_channels : int
        Number of channels in the skip connection.
    out_channels : int
        Number of output channels after fusion.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels + skip_channels, out_channels, dropout=dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Decoder feature map of shape [B, C, H, W].
        skip : torch.Tensor
            Skip feature map from the encoder.

        Returns
        -------
        torch.Tensor
            Refined decoder feature map.
        """
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SimpleUNetDecoder(nn.Module):
    """
    Custom U-Net-like decoder built on top of encoder feature maps.

    Parameters
    ----------
    encoder_channels : Sequence[int]
        Channel dimensions of encoder outputs.
    decoder_channels : Sequence[int]
        Output channels for decoder stages.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        dropout: float = 0.5,
    ):
        super().__init__()

        # Typical encoder outputs include features from shallow -> deep.
        # We reverse the usable part for decoder construction.
        # Example for depth=5: [3, 64, 64, 128, 256, 512]
        enc_ch = list(encoder_channels)

        # Use deepest feature as decoder input
        head_channels = enc_ch[-1]
        skip_channels = enc_ch[-2::-1]  # all remaining, reversed

        if len(decoder_channels) > len(skip_channels):
            raise ValueError(
                f"Too many decoder stages ({len(decoder_channels)}) for "
                f"available skip connections ({len(skip_channels)})."
            )

        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = skip_channels[: len(decoder_channels)]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, skip_ch, out_ch, dropout=dropout)
                for in_ch, skip_ch, out_ch in zip(
                    in_channels, skip_channels, decoder_channels
                )
            ]
        )

        self.out_channels = decoder_channels[-1]

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Decode encoder features into a high-resolution feature map.

        Parameters
        ----------
        features : Sequence[torch.Tensor]
            Feature maps returned by the encoder, ordered from shallow to deep.

        Returns
        -------
        torch.Tensor
            Final decoder feature map.
        """
        x = features[-1]
        skips = features[-2::-1]

        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)

        return x
