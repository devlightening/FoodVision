from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "backend" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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

DISPLAY_NAMES_TR = {
    "pizza": "Pizza",
    "hamburger": "Hamburger",
    "french_fries": "Patates Kızartması",
    "ice_cream": "Dondurma",
    "baklava": "Baklava",
    "donuts": "Donut",
    "omelette": "Omlet",
    "chicken_wings": "Tavuk Kanat",
    "steak": "Biftek / Et",
    "sushi": "Sushi",
}


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42


def main() -> None:
    print("Loading Food-101 from Hugging Face (ethz/food101)...")
    ds = load_dataset("ethz/food101")

    label_names = ds["train"].features["label"].names
    class_name_to_id = {name: idx for idx, name in enumerate(label_names)}
    selected_label_ids = [class_name_to_id[name] for name in SELECTED_CLASSES]

    original_id_to_new_id = {orig_id: new_id for new_id, orig_id in enumerate(selected_label_ids)}

    def keep_selected(example):
        return example["label"] in selected_label_ids

    train_ds = ds["train"].filter(keep_selected)
    val_ds = ds["validation"].filter(keep_selected)

    def remap_label(example):
        example["label"] = original_id_to_new_id[example["label"]]
        return example

    train_ds = train_ds.map(remap_label)
    val_ds = val_ds.map(remap_label)

    def preprocess(example):
        image = example["image"].convert("RGB").resize(IMAGE_SIZE)
        image = np.array(image).astype(np.float32) / 255.0
        return {"pixel_values": image, "labels": int(example["label"])}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

    def to_tf_dataset(hf_split, shuffle: bool):
        x = np.stack(hf_split["pixel_values"])
        y = np.array(hf_split["labels"], dtype=np.int64)
        tf_ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            tf_ds = tf_ds.shuffle(buffer_size=min(len(hf_split), 2000), seed=SEED)
        tf_ds = tf_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return tf_ds

    tf_train = to_tf_dataset(train_ds, shuffle=True)
    tf_val = to_tf_dataset(val_ds, shuffle=False)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(SELECTED_CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Starting training...")
    history = model.fit(tf_train, validation_data=tf_val, epochs=EPOCHS)

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    model_out = MODEL_DIR / "food_model.keras"
    model.save(model_out)
    print(f"Saved model to: {model_out}")

    id_to_class_name = {str(i): name for i, name in enumerate(SELECTED_CLASSES)}
    id_to_display_name_tr = {str(i): DISPLAY_NAMES_TR[name] for i, name in enumerate(SELECTED_CLASSES)}

    mapping = {
        "selected_classes": SELECTED_CLASSES,
        "id_to_class_name": id_to_class_name,
        "id_to_display_name_tr": id_to_display_name_tr,
    }
    mapping_out = MODEL_DIR / "class_mapping.json"
    mapping_out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved class mapping to: {mapping_out}")


if __name__ == "__main__":
    main()

