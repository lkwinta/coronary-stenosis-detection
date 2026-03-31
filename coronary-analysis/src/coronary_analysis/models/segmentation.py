from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder

from coronary_analysis.models.unet import SimpleUNetDecoder


class CoronaryDeeplabV3Plus(nn.Module):
    """
    DeepLabV3+ style segmentation model for coronary vessel segmentation.

    This model is a custom implementation of the DeepLabV3+ architecture,
    which features an encoder-decoder structure with atrous spatial pyramid pooling.

    Parameters
    ----------
    encoder_name : str, optional
        Name of the encoder backbone. Default is "resnet34".
    encoder_weights : str | None, optional
        Pretrained encoder weights, e.g. "imagenet". Default is "imagenet".

    Notes
    -----
    Input shape:
        [B, 1, H, W]
    Output shape:
        [B, 1, H, W] logits
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass of the DeepLabV3+ model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Raw segmentation logits of shape [B, 1, H, W].
        """
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-pixel vessel probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Probability map in [0, 1].
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict a binary vessel mask.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].
        threshold : float, optional
            Threshold applied to probabilities. Default is 0.5.

        Returns
        -------
        torch.Tensor
            Binary mask of shape [B, 1, H, W].
        """
        probs = self.predict_proba(x)
        return (probs > threshold).float()


class CoronaryUNetPP(nn.Module):
    """
    U-Net++ style segmentation model for coronary vessel segmentation.

    This model is a custom implementation of the U-Net++ architecture,
    which features nested skip connections and dense decoder blocks.

    Parameters
    ----------
    encoder_name : str, optional
        Name of the encoder backbone. Default is "resnet34".
    encoder_weights : str | None, optional
        Pretrained encoder weights, e.g. "imagenet". Default is "imagenet".
    decoder_channels : Sequence[int], optional
        Channel sizes for decoder stages. Default is (256, 128, 64, 32).
    depth : int, optional
        Encoder depth. Default is 5.

    Notes
    -----
    Input shape:
        [B, 1, H, W]
    Output shape:
        [B, 1, H, W] logits
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ):
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
            aux_params={
                "dropout": 0.5,
                "classes": 1,
                "pooling": "max",
                "activation": None,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass of the U-Net++ model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Raw segmentation logits of shape [B, 1, H, W].
        """
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-pixel vessel probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Probability map in [0, 1].
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict a binary vessel mask.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].
        threshold : float, optional
            Threshold applied to probabilities. Default is 0.5.

        Returns
        -------
        torch.Tensor
            Binary mask of shape [B, 1, H, W].
        """
        probs = self.predict_proba(x)
        return (probs > threshold).float()


class CoronaryUNet(nn.Module):
    """
    U-Net-based segmentation model for coronary angiography images.

    This module wraps a `segmentation_models_pytorch.Unet` model configured
    for single-channel grayscale input and binary segmentation output.

    The model is intended for vessel segmentation in coronary angiography,
    where:
    - input shape is `[B, 1, H, W]`
    - output shape is `[B, 1, H, W]`

    The forward pass returns raw logits. Helper methods are provided for:
    - converting logits to probabilities
    - thresholding probabilities into binary masks

    Parameters
    ----------
    encoder_name : str, optional
        Name of the encoder backbone used by the U-Net model.
        Examples include `"resnet34"`, `"resnet18"`, `"efficientnet-b0"`.
        Default is `"resnet34"`.
    encoder_weights : str, optional
        Pretrained weights used for the encoder backbone.
        Typically `"imagenet"` or `None`.
        Default is `"imagenet"`.

    Notes
    -----
    This model uses:
    - `in_channels=1` for grayscale angiography images
    - `classes=1` for binary segmentation

    The `forward()` method returns raw logits, which are suitable for use with
    losses such as `BCEWithLogitsLoss` or `DiceLoss(..., from_logits=True)`.
    """

    def __init__(
        self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet"
    ):
        """
        Initialize the coronary vessel segmentation model.

        Parameters
        ----------
        encoder_name : str, optional
            Name of the encoder backbone for the U-Net architecture.
        encoder_weights : str, optional
            Pretrained weights for the encoder backbone.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass of the segmentation model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `[B, 1, H, W]`.

        Returns
        -------
        torch.Tensor
            Raw segmentation logits of shape `[B, 1, H, W]`.

        Notes
        -----
        The returned tensor contains logits, not probabilities.
        To obtain probabilities, apply a sigmoid activation or use
        :meth:`predict_proba`.
        """
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-pixel foreground probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `[B, 1, H, W]`.

        Returns
        -------
        torch.Tensor
            Probability map of shape `[B, 1, H, W]`, with values in `[0, 1]`.

        Notes
        -----
        This method:
        1. switches the model to evaluation mode,
        2. computes logits using `forward()`,
        3. applies the sigmoid function to obtain probabilities.
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict a binary segmentation mask.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `[B, 1, H, W]`.
        threshold : float, optional
            Probability threshold used to convert the predicted probability map
            into a binary mask. Default is `0.5`.

        Returns
        -------
        torch.Tensor
            Binary mask tensor of shape `[B, 1, H, W]`, with values `0.0` or `1.0`.

        Notes
        -----
        This method first computes probabilities using :meth:`predict_proba`,
        then thresholds them to obtain a binary segmentation mask.
        """
        probs = self.predict_proba(x)
        return (probs > threshold).float()


class SegmentationHead(nn.Module):
    """
    Final segmentation head mapping decoder features to logits.

    Parameters
    ----------
    in_channels : int
        Number of input channels from the decoder.
    out_channels : int, optional
        Number of output channels. Default is 1 for binary segmentation.
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the segmentation head.

        Parameters
        ----------
        x : torch.Tensor
            Decoder feature map.

        Returns
        -------
        torch.Tensor
            Logits tensor.
        """
        return self.conv(x)


class CoronaryUNetCustom(nn.Module):
    """
    Coronary vessel segmentation model with explicit encoder, decoder, and head.

    This model uses:
    - a pretrained encoder from `segmentation_models_pytorch`
    - a custom U-Net-like decoder
    - a simple 1x1 convolution segmentation head

    Parameters
    ----------
    encoder_name : str, optional
        Name of the encoder backbone. Default is "resnet34".
    encoder_weights : str | None, optional
        Pretrained encoder weights, e.g. "imagenet". Default is "imagenet".
    decoder_channels : Sequence[int], optional
        Channel sizes for decoder stages.
    depth : int, optional
        Encoder depth. Default is 5.

    Notes
    -----
    Input shape:
        [B, 1, H, W]

    Output shape:
        [B, 1, H, W] logits
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        depth: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=1,
            depth=depth,
            weights=encoder_weights,
        )

        self.decoder = SimpleUNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full segmentation model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Raw segmentation logits of shape [B, 1, H, W].
        """
        features = self.encoder(x)
        decoded = self.decoder(features)
        logits = self.segmentation_head(decoded)

        # Optionally upsample to exactly match input size
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict per-pixel vessel probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Probability map in [0, 1].
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict a binary vessel mask.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, H, W].
        threshold : float, optional
            Threshold applied to probabilities. Default is 0.5.

        Returns
        -------
        torch.Tensor
            Binary mask of shape [B, 1, H, W].
        """
        probs = self.predict_proba(x)
        return (probs > threshold).float()
