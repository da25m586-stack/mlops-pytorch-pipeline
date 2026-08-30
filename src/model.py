import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(architecture: str = "resnet18", num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if architecture == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported architecture: {architecture}")
