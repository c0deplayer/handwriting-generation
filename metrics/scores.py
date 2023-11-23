from torchvision.transforms import Compose, ToTensor

from .backbones import VGG16Backbone
from .base_score import BaseScore
from .distances import EuclideanDistance
from .transforms import PaddingMin, ResizeHeight

VGG16_10400_URL = "https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth"


class HWDScore(BaseScore):
    def __init__(self, device="cpu"):
        backbone = VGG16Backbone(VGG16_10400_URL, device=device)
        distance = EuclideanDistance()
        transforms = Compose(
            [
                ResizeHeight(32),
                ToTensor(),
                PaddingMin(32, 32),
            ]
        )
        super().__init__(backbone, distance, transforms)
