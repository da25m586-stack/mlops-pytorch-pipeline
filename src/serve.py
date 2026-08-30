import io
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

from model import get_model
from dataset import CIFAR10_CLASSES

app = Flask(__name__)

device = "cpu"
model = None

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    global model
    checkpoint_path = Path(os.environ.get("MODEL_PATH", "checkpoints/best_model.pt"))
    architecture = os.environ.get("MODEL_ARCH", "resnet18")
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))

    m = get_model(architecture=architecture, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(device)
    m.eval()
    model = m


@app.get("/health")
def health():
    if model is None:
        return jsonify({"status": "not_ready"}), 503
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "missing 'image' field"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return jsonify({
        "probabilities": {cls: round(p, 6) for cls, p in zip(CIFAR10_CLASSES, probs)},
        "predicted_class": CIFAR10_CLASSES[int(torch.argmax(torch.tensor(probs)).item())],
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)
