from __future__ import annotations

import csv
import json
import math
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.h5"
DATASET_ROOT = BASE_DIR / "final_dataset"
TEST_DATA_PATH = DATASET_ROOT / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def configure_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D6DCE5",
            "axes.labelcolor": "#1F2937",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#D9E2EC",
            "grid.alpha": 0.6,
            "font.size": 10,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D6DCE5",
        }
    )


def resolve_class_labels(class_indices: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def discover_class_labels(test_data_path: Path) -> list[str]:
    class_labels = sorted(path.name for path in test_data_path.iterdir() if path.is_dir())
    if not class_labels:
        raise FileNotFoundError(f"No class folders found in: {test_data_path}")
    return class_labels


def make_readable_label(label: str) -> str:
    return label.replace("_", " ").replace(" ,", ",").strip()


def wrap_label(label: str, width: int = 22) -> str:
    return textwrap.fill(
        make_readable_label(label),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def count_images_in_directory(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS)


def collect_split_distribution(dataset_root: Path, class_labels: list[str]) -> dict[str, np.ndarray]:
    distribution: dict[str, np.ndarray] = {}

    for split_name in ("train", "val", "test"):
        split_dir = dataset_root / split_name
        split_counts = [count_images_in_directory(split_dir / class_name) for class_name in class_labels]
        distribution[split_name] = np.asarray(split_counts, dtype=int)

    return distribution


def collect_filepaths_and_labels(test_data_path: Path, class_labels: list[str]) -> tuple[list[str], np.ndarray]:
    filepaths: list[str] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_labels):
        class_dir = test_data_path / class_name
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                filepaths.append(str(file_path))
                labels.append(class_index)

    if not filepaths:
        raise FileNotFoundError(f"No image files found in: {test_data_path}")

    return filepaths, np.asarray(labels, dtype=np.int32)


def load_and_preprocess_image(file_path: tf.Tensor, label: tf.Tensor, num_classes: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=num_classes)


def build_evaluation_dataset(filepaths: list[str], labels: np.ndarray, num_classes: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics_overview(metrics: dict[str, float], output_path: Path) -> None:
    names = [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro F1",
    ]
    values = [
        metrics["accuracy"],
        metrics["weighted_precision"],
        metrics["weighted_recall"],
        metrics["weighted_f1"],
        metrics["macro_f1"],
    ]
    colors = ["#2563EB", "#0F766E", "#059669", "#7C3AED", "#EA580C"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="#1F2937", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Evaluation Metrics")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.99,
        0.03,
        f"Loss: {metrics['loss']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )

    save_figure(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, class_labels: list[str], output_path: Path, normalize: bool = False) -> None:
    num_classes = len(class_labels)
    figure_size = max(18, num_classes * 0.58)
    labels = [wrap_label(label, width=18) for label in class_labels]

    matrix = cm.astype(float)
    title = "Confusion Matrix (Counts)"
    colorbar_label = "Image Count"
    annotation_format = ".0f"

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        title = "Confusion Matrix (Row-Normalized)"
        colorbar_label = "Recall Ratio"
        annotation_format = ".2f"

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    if num_classes <= 15:
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_idx in range(num_classes):
            for col_idx in range(num_classes):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "#0F172A",
                    fontsize=8,
                )

    save_figure(fig, output_path)


def plot_top_confusions(cm: np.ndarray, class_labels: list[str], output_path: Path, limit: int = 20) -> list[dict[str, float | int | str]]:
    confusion_rows: list[dict[str, float | int | str]] = []

    for true_idx in range(cm.shape[0]):
        true_total = int(cm[true_idx].sum())
        for pred_idx in range(cm.shape[1]):
            count = int(cm[true_idx, pred_idx])
            if true_idx == pred_idx or count == 0:
                continue

            confusion_rows.append(
                {
                    "true_class": class_labels[true_idx],
                    "predicted_class": class_labels[pred_idx],
                    "count": count,
                    "true_class_total": true_total,
                    "true_class_error_rate": (count / true_total) if true_total else 0.0,
                }
            )

    confusion_rows.sort(key=lambda row: int(row["count"]), reverse=True)
    top_rows = confusion_rows[:limit]

    fig_height = max(6, len(top_rows) * 0.55)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    if top_rows:
        labels = [
            wrap_label(f"{row['true_class']} -> {row['predicted_class']}", width=40)
            for row in reversed(top_rows)
        ]
        counts = [int(row["count"]) for row in reversed(top_rows)]
        bars = ax.barh(labels, counts, color="#DC2626", edgecolor="#7F1D1D")
        ax.set_title("Top Misclassifications")
        ax.set_xlabel("Misclassified Images")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + (bar.get_height() / 2),
                str(count),
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax.text(0.5, 0.5, "No misclassifications found.", ha="center", va="center", fontsize=13)
        ax.axis("off")

    save_figure(fig, output_path)
    return confusion_rows


def plot_classwise_metrics(report_dict: dict[str, dict[str, float] | float], class_labels: list[str], output_path: Path) -> None:
    metric_matrix = np.asarray(
        [
            [
                float(report_dict[class_name]["precision"]),
                float(report_dict[class_name]["recall"]),
                float(report_dict[class_name]["f1-score"]),
            ]
            for class_name in class_labels
        ]
    )
    support_values = [int(report_dict[class_name]["support"]) for class_name in class_labels]
    y_labels = [wrap_label(f"{class_name} ({support})", width=26) for class_name, support in zip(class_labels, support_values)]

    fig_height = max(12, len(class_labels) * 0.34)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    image = ax.imshow(metric_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Score")

    ax.set_title("Class-wise Precision, Recall, and F1")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Precision", "Recall", "F1 Score"])
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)

    if len(class_labels) <= 20:
        for row_idx in range(metric_matrix.shape[0]):
            for col_idx in range(metric_matrix.shape[1]):
                value = metric_matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#14532D" if value >= 0.75 else "#111827",
                )

    save_figure(fig, output_path)


def plot_split_distribution(class_labels: list[str], split_distribution: dict[str, np.ndarray], output_path: Path) -> None:
    y_positions = np.arange(len(class_labels))
    bar_height = 0.22
    colors = {"train": "#2563EB", "val": "#10B981", "test": "#F59E0B"}
    fig_height = max(12, len(class_labels) * 0.36)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for offset, split_name in zip((-bar_height, 0, bar_height), ("train", "val", "test")):
        ax.barh(
            y_positions + offset,
            split_distribution[split_name],
            height=bar_height,
            label=split_name.title(),
            color=colors[split_name],
            alpha=0.9,
        )

    ax.set_title("Dataset Distribution by Class and Split")
    ax.set_xlabel("Number of Images")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([wrap_label(label, width=24) for label in class_labels], fontsize=8)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    save_figure(fig, output_path)


def plot_prediction_distribution(
    class_labels: list[str],
    true_counts: np.ndarray,
    predicted_counts: np.ndarray,
    output_path: Path,
) -> None:
    y_positions = np.arange(len(class_labels))
    bar_height = 0.35
    fig_height = max(12, len(class_labels) * 0.36)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    ax.barh(y_positions - (bar_height / 2), true_counts, height=bar_height, label="True Count", color="#2563EB")
    ax.barh(y_positions + (bar_height / 2), predicted_counts, height=bar_height, label="Predicted Count", color="#F97316")

    ax.set_title("True vs Predicted Class Distribution (Test Set)")
    ax.set_xlabel("Number of Images")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([wrap_label(label, width=24) for label in class_labels], fontsize=8)
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    save_figure(fig, output_path)


def plot_confidence_distribution(confidences: np.ndarray, correctness: np.ndarray, output_path: Path) -> None:
    correct_confidences = confidences[correctness]
    incorrect_confidences = confidences[~correctness]

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 1, 21)

    if correct_confidences.size:
        ax.hist(correct_confidences, bins=bins, alpha=0.75, label="Correct Predictions", color="#16A34A", edgecolor="white")
    if incorrect_confidences.size:
        ax.hist(
            incorrect_confidences,
            bins=bins,
            alpha=0.75,
            label="Incorrect Predictions",
            color="#DC2626",
            edgecolor="white",
        )

    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Number of Images")
    ax.legend(loc="upper center")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    save_figure(fig, output_path)


def plot_sample_predictions(
    filepaths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    class_labels: list[str],
    output_path: Path,
    max_samples: int = 12,
) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    incorrect_indices = np.where(y_true != y_pred)[0]
    correct_indices = np.where(y_true == y_pred)[0]

    selected_indices: list[int] = []
    desired_incorrect = min(len(incorrect_indices), max_samples // 2)
    desired_correct = min(len(correct_indices), max_samples - desired_incorrect)

    if desired_incorrect:
        selected_indices.extend(rng.choice(incorrect_indices, size=desired_incorrect, replace=False).tolist())
    if desired_correct:
        selected_indices.extend(rng.choice(correct_indices, size=desired_correct, replace=False).tolist())

    if len(selected_indices) < min(max_samples, len(y_true)):
        remaining_pool = np.setdiff1d(np.arange(len(y_true)), np.asarray(selected_indices, dtype=int), assume_unique=False)
        extra_needed = min(max_samples, len(y_true)) - len(selected_indices)
        if extra_needed > 0 and len(remaining_pool) > 0:
            selected_indices.extend(rng.choice(remaining_pool, size=extra_needed, replace=False).tolist())

    selected_indices = sorted(selected_indices)
    if not selected_indices:
        return

    columns = 4
    rows = math.ceil(len(selected_indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, rows * 4.6))
    axes = np.atleast_1d(axes).ravel()

    for axis, sample_index in zip(axes, selected_indices):
        image = tf.keras.utils.load_img(filepaths[sample_index], target_size=IMAGE_SIZE)
        image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
        axis.imshow(image_array)

        predicted_label = class_labels[int(y_pred[sample_index])]
        true_label = class_labels[int(y_true[sample_index])]
        confidence = float(y_pred_probs[sample_index, y_pred[sample_index]])
        is_correct = predicted_label == true_label

        axis.set_title(
            f"Pred: {wrap_label(predicted_label, 18)}\n"
            f"True: {wrap_label(true_label, 18)}\n"
            f"Confidence: {confidence:.2%}",
            fontsize=9,
            color="#166534" if is_correct else "#991B1B",
        )
        axis.axis("off")

    for axis in axes[len(selected_indices) :]:
        axis.axis("off")

    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_path)


def save_class_summary_csv(
    class_labels: list[str],
    report_dict: dict[str, dict[str, float] | float],
    split_distribution: dict[str, np.ndarray],
    predicted_counts: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_name",
                "train_count",
                "val_count",
                "test_count",
                "predicted_count",
                "precision",
                "recall",
                "f1_score",
                "support",
            ]
        )

        for index, class_name in enumerate(class_labels):
            class_metrics = report_dict[class_name]
            writer.writerow(
                [
                    class_name,
                    int(split_distribution["train"][index]),
                    int(split_distribution["val"][index]),
                    int(split_distribution["test"][index]),
                    int(predicted_counts[index]),
                    float(class_metrics["precision"]),
                    float(class_metrics["recall"]),
                    float(class_metrics["f1-score"]),
                    int(class_metrics["support"]),
                ]
            )


def save_top_confusions_csv(confusion_rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "predicted_class", "count", "true_class_total", "true_class_error_rate"])
        for row in confusion_rows:
            writer.writerow(
                [
                    row["true_class"],
                    row["predicted_class"],
                    int(row["count"]),
                    int(row["true_class_total"]),
                    float(row["true_class_error_rate"]),
                ]
            )


def save_metrics_json(metrics: dict[str, float | int | str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metrics = {
        key: float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value
        for key, value in metrics.items()
    }
    output_path.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")


def build_text_report(
    model_name: str,
    model_path: Path,
    test_data_path: Path,
    metrics: dict[str, float | int | str],
    report_text: str,
    class_labels: list[str],
    split_distribution: dict[str, np.ndarray],
    confusion_rows: list[dict[str, float | int | str]],
    generated_files: list[str],
) -> str:
    split_totals = {split_name: int(counts.sum()) for split_name, counts in split_distribution.items()}
    low_support_classes = [
        f"{class_name} ({int(test_count)})"
        for class_name, test_count in zip(class_labels, split_distribution["test"])
        if int(test_count) < 100
    ]

    report_lines = [
        "Model Evaluation Report",
        "=" * 80,
        f"Generated At: {metrics['evaluated_at']}",
        f"Model Name: {model_name}",
        f"Model Path: {model_path}",
        f"Test Data Path: {test_data_path}",
        "",
        "Overall Metrics",
        "-" * 80,
        f"Loss:               {float(metrics['loss']):.6f}",
        f"Accuracy:           {float(metrics['accuracy']):.6f}",
        f"Weighted Precision: {float(metrics['weighted_precision']):.6f}",
        f"Weighted Recall:    {float(metrics['weighted_recall']):.6f}",
        f"Weighted F1 Score:  {float(metrics['weighted_f1']):.6f}",
        f"Macro Precision:    {float(metrics['macro_precision']):.6f}",
        f"Macro Recall:       {float(metrics['macro_recall']):.6f}",
        f"Macro F1 Score:     {float(metrics['macro_f1']):.6f}",
        f"Test Images:        {int(metrics['test_images'])}",
        f"Number of Classes:  {int(metrics['num_classes'])}",
        "",
        "Dataset Split Totals",
        "-" * 80,
        f"Train Images: {split_totals['train']}",
        f"Val Images:   {split_totals['val']}",
        f"Test Images:  {split_totals['test']}",
        "",
        "Low-Support Classes (< 100 images in a class during evaluation)",
        "-" * 80,
    ]

    if low_support_classes:
        report_lines.extend(low_support_classes)
    else:
        report_lines.append("None")

    report_lines.extend(
        [
            "",
            "Top Misclassifications",
            "-" * 80,
        ]
    )

    if confusion_rows:
        for index, row in enumerate(confusion_rows[:15], start=1):
            report_lines.append(
                f"{index:>2}. {row['true_class']} -> {row['predicted_class']} | "
                f"count={int(row['count'])}, error_rate={float(row['true_class_error_rate']):.2%}"
            )
    else:
        report_lines.append("No misclassifications found.")

    report_lines.extend(
        [
            "",
            "Classification Report",
            "-" * 80,
            report_text.strip(),
            "",
            "Saved Files",
            "-" * 80,
        ]
    )
    report_lines.extend(sorted(generated_files))
    return "\n".join(report_lines) + "\n"


def main() -> None:
    configure_plot_style()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {TEST_DATA_PATH}")

    model_name = MODEL_PATH.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = BASE_DIR / f"{model_name}_{timestamp}"
    result_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(str(MODEL_PATH), compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"Loading test data from: {TEST_DATA_PATH}")
    class_labels = discover_class_labels(TEST_DATA_PATH)
    filepaths, y_true = collect_filepaths_and_labels(TEST_DATA_PATH, class_labels)
    test_dataset = build_evaluation_dataset(filepaths, y_true, num_classes=len(class_labels))
    print(f"Discovered {len(filepaths)} test images across {len(class_labels)} classes.")

    split_distribution = collect_split_distribution(DATASET_ROOT, class_labels)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_dataset, verbose=1)

    print("Running predictions...")
    y_pred_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    predicted_counts = np.bincount(y_pred, minlength=len(class_labels))
    true_counts = np.bincount(y_true, minlength=len(class_labels))
    confidences = np.max(y_pred_probs, axis=1)
    correctness = y_true == y_pred

    metrics = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "test_images": int(len(y_true)),
        "num_classes": int(len(class_labels)),
    }

    print("Saving plots and reports...")
    generated_files = [
        "class_summary.csv",
        "class_wise_metrics.png",
        "confidence_distribution.png",
        "confusion_matrix_counts.png",
        "confusion_matrix_normalized.png",
        "dataset_distribution.png",
        "metrics_overview.png",
        "metrics_summary.json",
        "prediction_distribution.png",
        "report.txt",
        "sample_predictions.png",
        "top_confusions.csv",
        "top_confusions.png",
    ]

    plot_metrics_overview(metrics, result_folder / "metrics_overview.png")
    plot_confusion_matrix(cm, class_labels, result_folder / "confusion_matrix_counts.png", normalize=False)
    plot_confusion_matrix(cm, class_labels, result_folder / "confusion_matrix_normalized.png", normalize=True)
    confusion_rows = plot_top_confusions(cm, class_labels, result_folder / "top_confusions.png")
    plot_classwise_metrics(report_dict, class_labels, result_folder / "class_wise_metrics.png")
    plot_split_distribution(class_labels, split_distribution, result_folder / "dataset_distribution.png")
    plot_prediction_distribution(class_labels, true_counts, predicted_counts, result_folder / "prediction_distribution.png")
    plot_confidence_distribution(confidences, correctness, result_folder / "confidence_distribution.png")
    plot_sample_predictions(
        filepaths=filepaths,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_probs=y_pred_probs,
        class_labels=class_labels,
        output_path=result_folder / "sample_predictions.png",
    )

    save_class_summary_csv(
        class_labels=class_labels,
        report_dict=report_dict,
        split_distribution=split_distribution,
        predicted_counts=predicted_counts,
        output_path=result_folder / "class_summary.csv",
    )
    save_top_confusions_csv(confusion_rows, result_folder / "top_confusions.csv")
    save_metrics_json(metrics, result_folder / "metrics_summary.json")

    report_output = build_text_report(
        model_name=model_name,
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        metrics=metrics,
        report_text=report_text,
        class_labels=class_labels,
        split_distribution=split_distribution,
        confusion_rows=confusion_rows,
        generated_files=generated_files,
    )
    (result_folder / "report.txt").write_text(report_output, encoding="utf-8")

    print(f"\nAll results saved in folder: {result_folder}")
    for file_name in generated_files:
        print(f" - {file_name}")


if __name__ == "__main__":
    main()
