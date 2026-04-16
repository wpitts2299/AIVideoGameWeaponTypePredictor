"""Builds merged train/validation JSON datasets and dataset summary stats."""

import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


# Central file locations and label mappings for the two source datasets.
ROOT = Path(__file__).resolve().parent
WP1_XML = ROOT / "WPDataSet1" / "annotations" / "annotations cvat.xml"
WP2_METADATA = ROOT / "WPDataSet2" / "metadata.csv"
WP2_BASE = ROOT / "WPDataSet2" / "weapon_detection"
TRAIN_JSON = ROOT / "train.json"
VAL_JSON = ROOT / "val.json"
DATASET_STATS_JSON = ROOT / "dataset_stats.json"
RANDOM_SEED = 42
VAL_RATIO = 0.2

WP2_TARGET_TO_LABEL = {
    0: "Automatic Rifle",
    1: "Bazooka",
    2: "Grenade Launcher",
    3: "Handgun",
    4: "Knife",
    5: "Shotgun",
    6: "SMG",
    7: "Sniper",
    8: "Sword",
}

WP1_WEAPON_LABEL_MAP = {
    "pistol": "Handgun",
    "pistolet": "Handgun",
    "handgun": "Handgun",
    "automatic rifle": "Automatic Rifle",
    "shotgun": "Shotgun",
    "smg": "SMG",
    "sniper": "Sniper",
    "sword": "Sword",
    "knife": "Knife",
    "bazooka": "Bazooka",
    "grenade launcher": "Grenade Launcher",
}

WP1_IGNORED_TAGS = {
    "other weapons",
    "not weapon",
    "undetected",
    "undetectedt",
    "weapons are visible",
    "weapons can be seen, but it is seen very poorly",
    "nothing visible",
}


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def normalize_wp1_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if key in WP1_IGNORED_TAGS:
        return None
    return WP1_WEAPON_LABEL_MAP.get(key)


def stratified_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    # WPDataSet1 does not come with a train/val split, so create one per class.
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    train_records: list[dict] = []
    val_records: list[dict] = []

    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)

        if len(items) <= 1:
            val_count = 0
        else:
            val_count = max(1, int(round(len(items) * val_ratio)))
            val_count = min(val_count, len(items) - 1)

        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    return train_records, val_records


def load_wpdataset1() -> tuple[list[dict], list[dict], dict]:
    if not WP1_XML.exists():
        raise FileNotFoundError(f"Missing dataset file: {WP1_XML}")

    # WPDataSet1 stores useful labels in CVAT image tags rather than COCO boxes.
    root = ET.parse(WP1_XML).getroot()
    parsed_records: list[dict] = []
    skipped = Counter()

    for image_node in root.findall("image"):
        image_name = image_node.attrib["name"]
        image_path = ROOT / "WPDataSet1" / image_name
        raw_tags = [tag.attrib.get("label", "").strip() for tag in image_node.findall("tag")]
        weapon_labels = sorted(
            {
                normalized
                for normalized in (normalize_wp1_label(tag) for tag in raw_tags)
                if normalized is not None
            }
        )

        if not image_path.exists():
            skipped["missing_image"] += 1
            continue
        if not weapon_labels:
            skipped["no_supported_weapon_label"] += 1
            continue
        if len(weapon_labels) > 1:
            skipped["ambiguous_weapon_labels"] += 1
            continue

        parsed_records.append(
            {
                "image_path": rel_path(image_path),
                "label": weapon_labels[0],
                "source": "WPDataSet1",
            }
        )

    train_records, val_records = stratified_split(parsed_records, VAL_RATIO, RANDOM_SEED)
    return train_records, val_records, dict(skipped)


def infer_wp2_label_from_filename(image_name: str) -> str:
    return Path(image_name).stem.rsplit("_", 1)[0]


def load_wpdataset2() -> tuple[list[dict], list[dict], dict]:
    if not WP2_METADATA.exists():
        raise FileNotFoundError(f"Missing dataset file: {WP2_METADATA}")

    # WPDataSet2 already ships with a split in metadata.csv.
    train_records: list[dict] = []
    val_records: list[dict] = []
    skipped = Counter()

    with WP2_METADATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = "train" if row["train_id"] == "1" else "val"
            image_name = row["imagefile"]
            image_path = WP2_BASE / split / "images" / image_name

            if not image_path.exists():
                skipped["missing_image"] += 1
                continue

            target = int(row["target"])
            target_label = WP2_TARGET_TO_LABEL.get(target)
            inferred_label = infer_wp2_label_from_filename(image_name)

            if target_label is None:
                skipped["unknown_target"] += 1
                continue
            if inferred_label != target_label:
                skipped["target_filename_mismatch"] += 1
                continue

            record = {
                "image_path": rel_path(image_path),
                "label": target_label,
                "source": "WPDataSet2",
            }
            if split == "train":
                train_records.append(record)
            else:
                val_records.append(record)

    return train_records, val_records, dict(skipped)


def class_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(record["label"] for record in records)
    return dict(sorted(counts.items()))


def source_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(record["source"] for record in records)
    return dict(sorted(counts.items()))


def strip_internal_fields(records: list[dict]) -> list[dict]:
    return [{"image_path": record["image_path"], "label": record["label"]} for record in records]


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_stats(train_records: list[dict], val_records: list[dict], skipped: dict) -> dict:
    # Keep a compact summary alongside the generated JSON splits for quick inspection.
    all_records = train_records + val_records
    return {
        "random_seed": RANDOM_SEED,
        "val_ratio_for_wpdataset1": VAL_RATIO,
        "total_samples": len(all_records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "classes": sorted({record["label"] for record in all_records}),
        "class_counts": {
            "train": class_counts(train_records),
            "val": class_counts(val_records),
            "all": class_counts(all_records),
        },
        "source_counts": {
            "train": source_counts(train_records),
            "val": source_counts(val_records),
            "all": source_counts(all_records),
        },
        "skipped": skipped,
    }


def main() -> None:
    # Merge both sources into one consistent label space and write the training artifacts.
    wp1_train, wp1_val, wp1_skipped = load_wpdataset1()
    wp2_train, wp2_val, wp2_skipped = load_wpdataset2()

    train_records = wp1_train + wp2_train
    val_records = wp1_val + wp2_val

    stats = build_stats(
        train_records=train_records,
        val_records=val_records,
        skipped={
            "WPDataSet1": wp1_skipped,
            "WPDataSet2": wp2_skipped,
        },
    )

    save_json(TRAIN_JSON, strip_internal_fields(train_records))
    save_json(VAL_JSON, strip_internal_fields(val_records))
    save_json(DATASET_STATS_JSON, stats)

    print(f"Saved {TRAIN_JSON.name} with {len(train_records)} samples")
    print(f"Saved {VAL_JSON.name} with {len(val_records)} samples")
    print(f"Saved {DATASET_STATS_JSON.name}")
    print("\nClass counts:")
    for label, count in stats["class_counts"]["all"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
