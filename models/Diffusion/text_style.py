import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torchvision import models

from .attention import AffineTransformLayer, MultiHeadAttention
from .cnn import FeedForwardNetwork
from .utils import reshape_up


class StyleExtractor(nn.Module):
    def __init__(self, device: torch.device) -> None:
        """
        Takes a grayscale image (with the last channel) with pixels [0, 255].
        Rescales to [-1, 1] and repeats along the channel axis for 3 channels.
        Uses a MobileNetV2 with pretrained model_checkpoints from imagenet as initial model_checkpoints.

        Parameters
        ----------
        device: torch.device
            _description_
        """
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
        """_summary_

        Args:
            image (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        x = image.clone().detach().to(self.device)

        x = (2 * x / 255) - 1
        x = repeat(x, "1 1 h w -> 1 3 h w")

        x = self.mobilenet_v2(x)
        x = self.pooling(x)
        x = rearrange(x, "1 h 1 w -> 1 h w")

        return x.cpu()


class TextStyleEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int,
        vocab_size: int = 73,
        hidden_size: int = 512,
    ) -> None:
        """_summary_

        Parameters
        ----------
        in_features: int
            _description_
        d_model: int
            _description_
        vocab_size: int
            _description_ by default 73
        hidden_size: int
            _description_, by default 512
        """
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.style_ffn = FeedForwardNetwork(
            in_features, d_model, hidden_size=hidden_size
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.affine_0 = AffineTransformLayer(in_features // 8, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.affine_1 = AffineTransformLayer(in_features // 8, d_model)

        self.mha = MultiHeadAttention(d_model, 8)
        self.affine_2 = AffineTransformLayer(in_features // 8, d_model)
        self.text_ffn = FeedForwardNetwork(d_model, d_model, hidden_size=d_model * 2)
        self.affine_3 = AffineTransformLayer(in_features // 8, d_model)

    def forward(self, text: Tensor, style: Tensor, sigma: Tensor) -> Tensor:
        """
        _summary_

        Parameters
        ----------
        text: Tensor
            _description_
        style: Tensor
            _description_
        sigma: Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        
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
