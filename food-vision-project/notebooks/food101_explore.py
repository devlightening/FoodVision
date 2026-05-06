from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


SELECTED_CLASSES = [
    "pizza",
    "hamburger",
    "french_fries",
    "ice_cream",
    "baklava",
    "donuts",
    "omelette",
    "chicken_wings",
    "steak",
    "sushi",
]


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Food-101 from Hugging Face (ethz/food101)...")
    ds = load_dataset("ethz/food101")
    print("Food-101 dataset loaded")

    label_names = ds["train"].features["label"].names
    print(f"101 classes found: {len(label_names)}")
    print("All label names:")
    for i, name in enumerate(label_names):
        print(f"{i:3d}: {name}")

    missing = [c for c in SELECTED_CLASSES if c not in label_names]
    if missing:
        raise ValueError(f"Selected classes missing from dataset: {missing}")
    print("Selected 10 classes verified")

    class_name_to_id = {name: idx for idx, name in enumerate(label_names)}
    selected_label_ids = [class_name_to_id[name] for name in SELECTED_CLASSES]
    print("Selected class name -> label id:")
    for name in SELECTED_CLASSES:
        print(f"- {name}: {class_name_to_id[name]}")

    def keep_selected(example):
        return example["label"] in selected_label_ids

    train_filtered = ds["train"].filter(keep_selected)
    val_filtered = ds["validation"].filter(keep_selected)

    print(f"Filtered train count: {len(train_filtered)}")
    print(f"Filtered validation count: {len(val_filtered)}")

    def per_class_counts(split):
        counts = Counter(split["label"])
        return {label_names[k]: int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}

    train_counts = per_class_counts(train_filtered)
    val_counts = per_class_counts(val_filtered)

    print("Per-class distribution (train):")
    for name in SELECTED_CLASSES:
        print(f"- {name}: {train_counts.get(name, 0)}")
    print("Per-class distribution (validation):")
    for name in SELECTED_CLASSES:
        print(f"- {name}: {val_counts.get(name, 0)}")

    # Collect 2 examples per selected class from the filtered train split.
    examples_by_class = defaultdict(list)
    for ex in train_filtered:
        class_name = label_names[ex["label"]]
        if class_name in SELECTED_CLASSES and len(examples_by_class[class_name]) < 2:
            examples_by_class[class_name].append(ex)
        if all(len(examples_by_class[c]) >= 2 for c in SELECTED_CLASSES):
            break

    fig, axes = plt.subplots(nrows=len(SELECTED_CLASSES), ncols=2, figsize=(8, 4 * len(SELECTED_CLASSES)))
    for row, class_name in enumerate(SELECTED_CLASSES):
        for col in range(2):
            ax = axes[row][col]
            ax.axis("off")
            if col == 0:
                ax.set_title(class_name, fontsize=12, loc="left")
            if col < len(examples_by_class[class_name]):
                image = examples_by_class[class_name][col]["image"]
                ax.imshow(image)
    plt.tight_layout()

    out_path = DATASET_DIR / "selected_food101_samples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Sample image saved to {out_path}")


if __name__ == "__main__":
    main()

