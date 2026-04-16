"""Trains the weapon classifier and saves a checkpoint plus training metrics.

This file supports three training modes:
1. Standard split mode: train on train.json and validate on val.json.
2. Cross-validation mode: rotate the held-out fold across all folds and report fold results.
3. Seeded single-fold mode: build one reproducible 5-fold split, train on 4 folds, test on 1 fold.
"""

import argparse
import copy
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

# Default training configuration and output paths.
TRAIN_JSON = "train.json"
VAL_JSON = "val.json"
TRAINING_STATS_JSON = "training_stats.json"
MODEL_PATH = "weapon_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 1e-4
IMAGE_SIZE = 224
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FullDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


# Augment only the training loader; keep validation deterministic.
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

criterion = nn.CrossEntropyLoss()
weights = ViT_B_16_Weights.DEFAULT


def load_splits():
    # Load the prepared JSON splits and convert string labels to model class indices.
    missing_files = [path for path in (TRAIN_JSON, VAL_JSON) if not os.path.exists(path)]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing required dataset file(s): {missing_list}. "
            "Run prepare_dataset.py to create JSON files with "
            "[{'image_path': '...', 'label': '...'}] entries before training."
        )

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    if not train_data:
        raise ValueError("No training samples were found in train.json.")
    if not val_data:
        raise ValueError("No validation samples were found in val.json.")

    all_classes = sorted({item["label"] for item in train_data + val_data})
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

    def index_records(records):
        indexed = []
        for item in records:
            indexed.append(
                {
                    "image_path": item["image_path"],
                    "label": class_to_idx[item["label"]],
                }
            )
        return indexed

    return index_records(train_data), index_records(val_data), all_classes, train_data, val_data


def load_all_data():
    # CV and seeded fold modes use the combined dataset, not the fixed train/val split.
    train_data, val_data, classes, train_raw, val_raw = load_splits()
    return train_data + val_data, classes, train_raw + val_raw


def train_epoch(model, optimizer, train_loader):
    # Standard supervised training loop for one pass over the training loader.
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return loss_sum / total, correct / total


def predict_all(model, data_loader):
    # Collect raw predictions for metric calculation without updating weights.
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return all_labels, all_preds


def compute_metrics(all_labels, all_preds, classes):
    # Return both overall metrics and richer artifacts for later inspection.
    accuracy = sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(all_labels, all_preds).tolist()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def label_counts(records):
    counts = Counter(item["label"] for item in records)
    return dict(sorted(counts.items()))


def create_model(num_classes):
    # Replace the ViT classification head to match the project label set.
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def train_standard(args):
    # Plainest mode: use the already prepared train.json for training
    # and val.json for testing/validation.
    train_data, val_data, classes, train_raw, val_raw = load_splits()
    train_dataset = FullDataset(train_data, transform=train_transform)
    val_dataset = FullDataset(val_data, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    epoch_history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        val_labels, val_preds = predict_all(model, val_loader)
        val_metrics = compute_metrics(val_labels, val_preds, classes)

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        epoch_history.append(epoch_summary)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "classes": classes,
                },
                args.model_path,
            )
            print(f"Saved best model to {args.model_path}")

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state_dict)
    best_labels, best_preds = predict_all(model, val_loader)
    best_metrics = compute_metrics(best_labels, best_preds, classes)

    training_stats = {
        "mode": "train_val_split",
        "device": str(DEVICE),
        "random_seed": args.seed,
        "num_classes": len(classes),
        "classes": classes,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_class_counts": label_counts(train_raw),
        "val_class_counts": label_counts(val_raw),
        "best_epoch": best_epoch,
        "best_val_metrics": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
        },
        "classification_report": best_metrics["classification_report"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "epoch_history": epoch_history,
    }

    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    print("\nBest validation metrics:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1: {best_metrics['f1']:.4f}")
    print(f"Saved training stats to {args.stats_path}")
    print("Training finished")


def train_cross_validated(args):
    # Full cross-validation mode: split the whole dataset into N folds,
    # then repeat training N times so each fold gets one turn as the test fold.
    # This is mainly for measuring stability and average performance.
    all_data, classes, raw_records = load_all_data()
    base_dataset = FullDataset(all_data)
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    best_acc = 0.0
    best_fold = 0
    best_epoch = 0
    best_state_dict = None
    fold_results = []

    for fold_index, (train_idx, val_idx) in enumerate(kf.split(base_dataset.data), start=1):
        print(f"\nFold {fold_index}/{args.cv_folds}")

        train_subset = Subset(FullDataset(all_data, transform=train_transform), train_idx)
        val_subset = Subset(FullDataset(all_data, transform=val_transform), val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        model = create_model(len(classes))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        fold_best = None
        fold_epoch_history = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, optimizer, train_loader)
            val_labels, val_preds = predict_all(model, val_loader)
            val_metrics = compute_metrics(val_labels, val_preds, classes)

            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
            fold_epoch_history.append(epoch_summary)

            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            if fold_best is None or val_metrics["accuracy"] > fold_best["metrics"]["accuracy"]:
                fold_best = {
                    "epoch": epoch + 1,
                    "metrics": val_metrics,
                }

                if val_metrics["accuracy"] > best_acc:
                    best_acc = val_metrics["accuracy"]
                    best_fold = fold_index
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "model_state_dict": best_state_dict,
                            "classes": classes,
                        },
                        args.model_path,
                    )
                    print(f"Saved best model to {args.model_path}")

        fold_results.append(
            {
                "fold": fold_index,
                "num_train_samples": len(train_idx),
                "num_val_samples": len(val_idx),
                "best_epoch": fold_best["epoch"],
                "best_val_metrics": {
                    "accuracy": fold_best["metrics"]["accuracy"],
                    "precision": fold_best["metrics"]["precision"],
                    "recall": fold_best["metrics"]["recall"],
                    "f1": fold_best["metrics"]["f1"],
                },
                "classification_report": fold_best["metrics"]["classification_report"],
                "confusion_matrix": fold_best["metrics"]["confusion_matrix"],
                "epoch_history": fold_epoch_history,
            }
        )

    if best_state_dict is None:
        raise RuntimeError("Cross-validation did not produce a model checkpoint.")

    avg_accuracy = sum(item["best_val_metrics"]["accuracy"] for item in fold_results) / len(fold_results)
    avg_precision = sum(item["best_val_metrics"]["precision"] for item in fold_results) / len(fold_results)
    avg_recall = sum(item["best_val_metrics"]["recall"] for item in fold_results) / len(fold_results)
    avg_f1 = sum(item["best_val_metrics"]["f1"] for item in fold_results) / len(fold_results)

    training_stats = {
        "mode": "cross_validation",
        "device": str(DEVICE),
        "random_seed": args.seed,
        "cv_folds": args.cv_folds,
        "epochs": args.epochs,
        "num_classes": len(classes),
        "classes": classes,
        "total_samples": len(all_data),
        "class_counts": label_counts(raw_records),
        "best_overall_fold": best_fold,
        "best_overall_epoch": best_epoch,
        "best_overall_accuracy": best_acc,
        "average_best_fold_metrics": {
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        },
        "fold_results": fold_results,
    }

    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    print("\nCross-validation results:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Best fold checkpoint saved to {args.model_path}")
    print(f"Saved training stats to {args.stats_path}")
    print("Training finished")


def train_seeded_fold_split(args):
    # Single seeded fold mode: create the folds once with a fixed random seed,
    # choose one fold as the held-out test set, and train on the other folds.
    # In plain terms, this is "4 sets for training, 1 set for testing."
    all_data, classes, raw_records = load_all_data()
    base_dataset = FullDataset(all_data)
    kf = KFold(n_splits=args.fold_count, shuffle=True, random_state=args.seed)
    splits = list(kf.split(base_dataset.data))

    if args.test_fold < 1 or args.test_fold > len(splits):
        raise ValueError(f"--test-fold must be between 1 and {len(splits)}.")

    train_idx, val_idx = splits[args.test_fold - 1]
    train_subset = Subset(FullDataset(all_data, transform=train_transform), train_idx)
    val_subset = Subset(FullDataset(all_data, transform=val_transform), val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = create_model(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    epoch_history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        val_labels, val_preds = predict_all(model, val_loader)
        val_metrics = compute_metrics(val_labels, val_preds, classes)

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        epoch_history.append(epoch_summary)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "classes": classes,
                },
                args.model_path,
            )
            print(f"Saved best model to {args.model_path}")

    if best_state_dict is None:
        raise RuntimeError("Seeded fold-split training did not produce a model checkpoint.")

    model.load_state_dict(best_state_dict)
    best_labels, best_preds = predict_all(model, val_loader)
    best_metrics = compute_metrics(best_labels, best_preds, classes)

    training_stats = {
        "mode": "seeded_fold_split",
        "device": str(DEVICE),
        "random_seed": args.seed,
        "fold_count": args.fold_count,
        "test_fold": args.test_fold,
        "epochs": args.epochs,
        "num_classes": len(classes),
        "classes": classes,
        "total_samples": len(all_data),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "class_counts": label_counts(raw_records),
        "best_epoch": best_epoch,
        "best_val_metrics": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
        },
        "classification_report": best_metrics["classification_report"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "epoch_history": epoch_history,
    }

    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(training_stats, f, indent=2)

    print("\nSeeded fold-split results:")
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(val_idx)}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1: {best_metrics['f1']:.4f}")
    print(f"Saved training stats to {args.stats_path}")
    print("Training finished")


def parse_args():
    # Choose the mode from the command line:
    # - no special flags: standard train.json / val.json training
    # - --cv-folds N: full N-fold cross-validation
    # - --single-fold-split: one reproducible split made from fold_count folds
    parser = argparse.ArgumentParser(description="Train the weapon classifier.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs. Default: {NUM_EPOCHS}.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help=f"Output checkpoint path. Default: {MODEL_PATH}.",
    )
    parser.add_argument(
        "--stats-path",
        default=TRAINING_STATS_JSON,
        help=f"Output training stats JSON path. Default: {TRAINING_STATS_JSON}.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Use K-fold cross-validation with this many folds. Set to 0 to use train/val split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for fold generation. Default: {RANDOM_SEED}.",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=5,
        help="Number of folds to generate for seeded fold split mode. Default: 5.",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=1,
        help="Which fold to use as the held-out test fold in seeded fold split mode. Default: 1.",
    )
    parser.add_argument(
        "--single-fold-split",
        action="store_true",
        help="Train on fold_count-1 folds and test on one held-out fold selected by --test-fold.",
    )
    return parser.parse_args()


def main():
    # Priority order:
    # 1. If --single-fold-split is set, use one seeded train/test fold split.
    # 2. Else if --cv-folds > 1, run full cross-validation.
    # 3. Otherwise, use the normal train.json / val.json split.
    args = parse_args()
    if args.single_fold_split:
        train_seeded_fold_split(args)
    elif args.cv_folds and args.cv_folds > 1:
        train_cross_validated(args)
    else:
        train_standard(args)


if __name__ == "__main__":
    main()
