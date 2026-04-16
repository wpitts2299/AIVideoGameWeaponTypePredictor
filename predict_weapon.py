"""Loads a saved checkpoint and predicts weapon class probabilities for one image."""

import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16

# Default checkpoint and runtime device for one-off predictions.
MODEL_PATH = "weapon_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same image normalization as validation/evaluation.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]

# Recreate the classifier head shape before loading the saved weights.
model = vit_b_16(weights=None)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

def predict(image_path):
    # Return sorted class probabilities so callers can inspect more than the top class.
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

    # Sort probabilities descending
    sorted_indices = probs.argsort()[::-1]
    sorted_probs = [(classes[i], probs[i] * 100) for i in sorted_indices]

    return sorted_probs


def main():
    # Small CLI wrapper for single-image inference from PowerShell or cmd.
    parser = argparse.ArgumentParser(description="Predict weapon class from an image.")
    parser.add_argument("image_path", help="Path to the image to classify.")
    args = parser.parse_args()

    sorted_probs = predict(args.image_path)
    print(f"Device: {DEVICE}")
    print(f"Top Prediction: {sorted_probs[0][0]} ({sorted_probs[0][1]:.2f}%)")
    print("\nAll Probabilities (from greatest to least):")
    for label, prob in sorted_probs:
        print(f"{label}: {prob:.2f}%")


if __name__ == "__main__":
    main()
