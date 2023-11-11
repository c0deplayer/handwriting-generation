import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torchvision import models

from .attention import AffineTransformLayer
from .utils import reshape_up, FeedForwardNetwork


class StyleExtractor(nn.Module):
    """
    Extract Style Features from a Grayscale Image.

    This module takes a grayscale image (with the last channel) with pixel values in
    the range [0, 255]. It rescales the image to the range [-1, 1] and replicates it
    along the channel axis to create a 3-channel image. The module then utilizes a
    MobileNetV2 architecture with pretrained weights from ImageNet as the initial
    feature extractor.

    Args:
        device (torch.device): The device to run the StyleExtractor on.
    """

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()

        self.device = device
        self.mobilenet_v2 = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT,
            progress=True,
        ).features.to(device)

        self.pooling = nn.AvgPool2d(kernel_size=3)

        for p in self.mobilenet_v2.parameters():
            p.requires_grad = False

    def forward(self, image: Tensor) -> Tensor:
        x = image.clone().detach().to(self.device)

        x = x / 127.5 - 1
        x = repeat(x, "1 1 h w -> 1 3 h w")

        x = self.mobilenet_v2(x)
        x = self.pooling(x)
        x = rearrange(x, "1 h 1 w -> w h")

        return x.cpu()


class TextStyleEncoder(nn.Module):
    """
    The TextStyleEncoder module takes input text and style features and encodes them into a style-aware representation.

    Args:
        in_features (int): The input feature dimension.
        d_model (int): The output dimension of the encoded style-aware representation.
        vocab_size (int): The vocabulary size for text embedding. Defaults to 73.
        hidden_size (int): The hidden size for the feedforward networks. Defaults to 512.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        vocab_size: int = 73,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(0.3)
        self.style_ffn = FeedForwardNetwork(
            in_features, d_model, hidden_size=hidden_size
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.affine_0 = AffineTransformLayer(in_features // 8, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.affine_1 = AffineTransformLayer(in_features // 8, d_model)

        self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.affine_2 = AffineTransformLayer(in_features // 8, d_model)
        self.text_ffn = FeedForwardNetwork(d_model, d_model, hidden_size=d_model * 2)
        self.affine_3 = AffineTransformLayer(in_features // 8, d_model)

    def forward(self, text: Tensor, style: Tensor, sigma: Tensor) -> Tensor:
        style = reshape_up(self.dropout(style), factor=5)

        style = self.layer_norm(self.style_ffn(style))
        style = self.affine_0(style, sigma)

        text = self.embedding(text)
        text = self.affine_1(self.layer_norm(text), sigma)

        mha, _ = self.mha(text, style, style)
        text = self.affine_2(self.layer_norm(text + mha), sigma)

        text = self.layer_norm(self.text_ffn(text))
        text = self.affine_3(text, sigma)

        return text
