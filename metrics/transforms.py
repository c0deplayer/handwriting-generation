import torch
from PIL import Image


class ResizeHeight:
    def __init__(self, height, interpolation=Image.NEAREST):
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        return img.resize((int(self.height * w / h), self.height), self.interpolation)


class PaddingMin:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        c, w, h = img.shape
        width = max(self.width, w)
        height = max(self.height, h)
        return torch.nn.functional.pad(
            img, (0, height - h, 0, width - w), mode="constant", value=0
        )
