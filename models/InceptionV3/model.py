import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionV3_M(nn.Module):
    def __init__(self, num_class: int, *, device: torch.device):
        super(InceptionV3_M, self).__init__()

        self.num_class = num_class
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.inception = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
        )

        self.__froze_and_change_layers__()
        self.inception = self.inception.to(device)

        print("Model details")
        print("================================")
        print(self.inception)
        print("================================")

    def __froze_and_change_layers__(self):
        for name, param in self.inception.named_parameters():
            if "Mixed_5" not in name:
                param.requires_grad = False

        self.inception.Mixed_6a = nn.Identity()
        self.inception.Mixed_6b = nn.Identity()
        self.inception.Mixed_6c = nn.Identity()
        self.inception.Mixed_6d = nn.Identity()
        self.inception.Mixed_6e = nn.Identity()
        self.inception.AuxLogits = nn.Identity()
        self.inception.Mixed_7a = nn.Identity()
        self.inception.Mixed_7b = nn.Identity()
        self.inception.Mixed_7c = nn.Identity()
        self.inception.fc = nn.Linear(288, self.num_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.inception(self.normalize(x))
