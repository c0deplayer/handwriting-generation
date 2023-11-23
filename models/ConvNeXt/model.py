import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models.convnext import LayerNorm2d


class ConvNeXt_M(nn.Module):
    def __init__(self, num_class: int, *, device: torch.device):
        super(ConvNeXt_M, self).__init__()

        self.num_class = num_class
        self.transforms = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        self.__froze_and_change_layers__()
        self.convnext = self.convnext.to(device)

        print("Model details")
        print("================================")
        print(self.convnext)
        print("================================")

    def __froze_and_change_layers__(self):
        for name, param in self.convnext.named_parameters():
            if "6" not in name:
                param.requires_grad = False

        self.convnext.features[7] = nn.Identity()

        self.convnext.classifier[-1] = nn.Linear(
            self.convnext.features[6][-1].out_channels, self.num_class
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convnext(self.transforms(x))
