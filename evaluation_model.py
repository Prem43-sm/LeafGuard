from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.h5"
DATASET_TEST_DIR = BASE_DIR / "final_dataset" / "test"
METRICS_PATH = BASE_DIR / "model_metrics.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_model() -> dict[str, float | int | str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not DATASET_TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {DATASET_TEST_DIR}")

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    generator = datagen.flow_from_directory(
        str(DATASET_TEST_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        interpolation="bilinear",
    )

    predictions = model.predict(generator, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = generator.classes

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "test_images": int(generator.samples),
        "num_classes": int(generator.num_classes),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print("Model evaluation completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Saved metrics to: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
