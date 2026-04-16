"""Evaluates a saved checkpoint on the validation split and writes metrics."""

import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Default evaluation inputs and outputs.
MODEL_PATH = "weapon_model.pth"
VAL_JSON = "val.json"
EVAL_STATS_JSON = "evaluation_stats.json"
BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the preprocessing used by validation and prediction.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, class_to_idx, transform=None):
        self.data = data
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        label = self.class_to_idx[item['label']]
        if self.transform:
            image = self.transform(image)
        return image, label


def parse_args():
    # Allow evaluation of any saved checkpoint without renaming files.
    parser = argparse.ArgumentParser(description="Evaluate a trained weapon classifier.")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help=f"Checkpoint path to evaluate. Default: {MODEL_PATH}.",
    )
    parser.add_argument(
        "--stats-path",
        default=EVAL_STATS_JSON,
        help=f"Output evaluation stats JSON path. Default: {EVAL_STATS_JSON}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Rebuild the model architecture, then load the saved weights and class order.
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    classes = checkpoint["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Run the model over the validation split and summarize its behavior.
    val_ds = EvalDataset(val_data, class_to_idx=class_to_idx, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    print(f"Model: {args.model_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nPer class metrics:")
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)

    report_dict = classification_report(
        all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True
    )
    matrix = confusion_matrix(all_labels, all_preds).tolist()

    # Persist the evaluation so models can be compared later without rerunning inference.
    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_samples": len(all_labels),
                "classification_report": report_dict,
                "confusion_matrix": matrix,
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation stats to {args.stats_path}")


if __name__ == "__main__":
    main()
