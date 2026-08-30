import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ResNet18Classifier, get_model


def test_resnet18_output_shape():
    model = ResNet18Classifier(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_resnet18_single_sample():
    model = ResNet18Classifier(num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_get_model_returns_correct_type():
    model = get_model(architecture="resnet18", num_classes=10)
    assert isinstance(model, ResNet18Classifier)


def test_get_model_unknown_architecture():
    with pytest.raises(ValueError):
        get_model(architecture="unknown_arch")


def test_model_gradients_flow():
    model = ResNet18Classifier(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    loss = model(x).sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_model_num_classes():
    for n in [5, 10, 100]:
        model = get_model(num_classes=n)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == n


