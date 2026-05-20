from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from typing import Any

import cv2
import matplotlib.pyplot as plt


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUTPUT_DIR = "dataset sample images"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize one sample image per disease class, grouped by plant."
    )
    parser.add_argument(
        "--dataset-path",
        default="final_dataset/test",
        help="Path to a dataset root that contains class folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated plant grids will be saved.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Number of rows in each plant grid.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of columns in each plant grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save images without opening matplotlib windows.",
    )
    return parser.parse_args()


def clean_label(text: str) -> str:
    """Convert raw folder tokens into readable title-cased labels."""
    cleaned = (
        text.replace("___", " ")
        .replace("__", " ")
        .replace("_", " ")
        .replace(",", " ")
        .strip()
    )
    cleaned = " ".join(cleaned.split())
    return cleaned.title()


def split_single_underscore_name(folder_name: str) -> tuple[str, str]:
    """
    Split names like:
        corn_(maize)_healthy
        cherry_(including_sour)_powdery_mildew
        pepper,_bell_healthy
    """
    tokens = folder_name.split("_")
    if not tokens:
        return folder_name, "Unknown"

    plant_tokens = [tokens[0]]
    index = 1

    # Keep qualifiers such as "(maize)" or "(including_sour)" with the plant.
    if index < len(tokens) and tokens[index].startswith("("):
        parenthesis_depth = 0
        while index < len(tokens):
            token = tokens[index]
            plant_tokens.append(token)
            parenthesis_depth += token.count("(")
            parenthesis_depth -= token.count(")")
            index += 1
            if parenthesis_depth <= 0 and ")" in token:
                break

    # Keep labels such as "pepper,_bell" together as one plant name.
    elif plant_tokens[0].endswith(",") and index < len(tokens):
        plant_tokens.append(tokens[index])
        index += 1

    # Handle simple two-word plant qualifiers such as "pepper_bell".
    elif index < len(tokens) and tokens[index].lower() in {"bell"}:
        plant_tokens.append(tokens[index])
        index += 1

    disease_tokens = tokens[index:] if index < len(tokens) else ["Unknown"]
    return "_".join(plant_tokens), "_".join(disease_tokens)


def parse_class_name(folder_name: str) -> tuple[str, str]:
    """
    Split a class folder into plant and disease names.

    Expected format:
        PlantName___DiseaseName

    A few fallback patterns are supported so the script is easier to reuse
    with slight naming variations.
    """
    if "___" in folder_name:
        plant_name, disease_name = folder_name.split("___", 1)
    elif "__" in folder_name:
        plant_name, disease_name = folder_name.split("__", 1)
    elif "_" in folder_name:
        plant_name, disease_name = split_single_underscore_name(folder_name)
    else:
        plant_name, disease_name = folder_name, "Unknown"

    return clean_label(plant_name), clean_label(disease_name)


def is_image_file(file_name: str) -> bool:
    """Return True if the filename looks like a supported image."""
    return os.path.splitext(file_name)[1].lower() in VALID_EXTENSIONS


def find_valid_image(image_paths: list[str]) -> str | None:
    """
    Return the first readable image path from the provided candidates.

    Corrupt or unreadable files are skipped safely.
    """
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is not None:
            return image_path

        print(f"Warning: Skipping unreadable image: {image_path}")

    return None


def load_dataset(dataset_path: str) -> dict[str, dict[str, Any]]:
    """
    Read all class folders and collect their metadata.

    Returns a dictionary like:
    {
        "Tomato___Early_blight": {
            "class_name": "Tomato___Early_blight",
            "plant_name": "Tomato",
            "disease_name": "Early Blight",
            "class_path": "...",
            "image_paths": [...],
        },
        ...
    }
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    dataset_info: dict[str, dict[str, Any]] = {}

    for entry_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, entry_name)
        if not os.path.isdir(class_path):
            continue

        plant_name, disease_name = parse_class_name(entry_name)
        image_paths = [
            os.path.join(class_path, file_name)
            for file_name in sorted(os.listdir(class_path))
            if is_image_file(file_name)
        ]

        dataset_info[entry_name] = {
            "class_name": entry_name,
            "plant_name": plant_name,
            "disease_name": disease_name,
            "class_path": class_path,
            "image_paths": image_paths,
        }

    if not dataset_info:
        raise ValueError(f"No class folders found inside: {dataset_path}")

    return dataset_info


def group_by_plant(dataset_info: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """
    Group classes by plant name.

    Returns:
    {
        "Tomato": [("Early Blight", "/path/to/class"), ("Late Blight", "/path/to/class")],
        "Potato": [...],
    }
    """
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for class_info in dataset_info.values():
        grouped[class_info["plant_name"]].append(
            (class_info["disease_name"], class_info["class_path"])
        )

    return {
        plant_name: sorted(disease_items, key=lambda item: item[0].lower())
        for plant_name, disease_items in sorted(grouped.items(), key=lambda item: item[0].lower())
    }


def get_sample_images(
    dataset_info: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Randomly select one readable sample image for each disease class.

    Duplicate paths are avoided explicitly, even though each class is expected
    to contain distinct image files.
    """
    rng = random.Random(seed)
    used_paths: set[str] = set()
    sample_images: dict[str, dict[str, Any]] = {}

    for class_name in sorted(dataset_info):
        class_info = dataset_info[class_name]
        candidate_paths = list(class_info["image_paths"])
        rng.shuffle(candidate_paths)

        valid_path = find_valid_image(candidate_paths)
        if valid_path is None:
            print(f"Warning: No valid sample image found for class: {class_name}")
            continue

        if valid_path in used_paths:
            remaining_paths = [path for path in candidate_paths if path not in used_paths]
            valid_path = find_valid_image(remaining_paths)

        if valid_path is None or valid_path in used_paths:
            print(f"Warning: Could not assign a unique sample image for class: {class_name}")
            continue

        used_paths.add(valid_path)
        sample_images[class_name] = {
            "plant_name": class_info["plant_name"],
            "disease_name": class_info["disease_name"],
            "image_path": valid_path,
        }

    return sample_images


def load_image_rgb(image_path: str) -> Any | None:
    """Read an image with OpenCV and convert BGR to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to read image: {image_path}")
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_grid(
    plant_name: str,
    plant_samples: list[dict[str, Any]],
    rows: int = 2,
    cols: int = 6,
) -> tuple[Any, Any]:
    """
    Create a matplotlib grid for one plant.

    The grid will show up to rows * cols images.
    """
    max_images = rows * cols
    selected_samples = sorted(plant_samples, key=lambda item: item["disease_name"].lower())[:max_images]
    figure_width = max(15, cols * 2.6)
    figure_height = max(8, rows * 4.0)

    fig, axes = plt.subplots(rows, cols, figsize=(figure_width, figure_height))

    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [axis for row_axes in axes for axis in row_axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, sample in zip(axes_list, selected_samples):
        image = load_image_rgb(sample["image_path"])
        if image is None:
            axis.text(
                0.5,
                0.5,
                "Unreadable Image",
                ha="center",
                va="center",
                fontsize=11,
                color="red",
            )
            continue

        axis.imshow(image)
        axis.set_title(
            f"{sample['plant_name']}\n{sample['disease_name']}",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        axis.axis("off")

    fig.suptitle(
        f"{plant_name} Disease Samples",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def save_output(fig: Any, output_path: str) -> None:
    """Save a matplotlib figure to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")


def safe_file_name(label: str) -> str:
    """Create a filesystem-safe file name from a plant label."""
    safe_name = "".join(ch if ch.isalnum() or ch in {" ", "-", "_"} else "_" for ch in label)
    return safe_name.strip().replace(" ", "_")


def organize_samples_by_plant(
    sample_images: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Convert flat class samples into a plant -> sample list mapping."""
    grouped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample_info in sample_images.values():
        grouped_samples[sample_info["plant_name"]].append(sample_info)

    return {
        plant_name: sorted(samples, key=lambda item: item["disease_name"].lower())
        for plant_name, samples in sorted(grouped_samples.items(), key=lambda item: item[0].lower())
    }


def main() -> None:
    """Run the dataset visualization workflow."""
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("Grid rows and columns must be positive integers.")

    dataset_path = os.path.abspath(args.dataset_path)
    output_dir = os.path.abspath(args.output_dir)
    max_images_per_plant = args.rows * args.cols

    # Step 1: Analyze the dataset and build the class metadata.
    dataset_info = load_dataset(dataset_path)
    plant_dictionary = group_by_plant(dataset_info)

    print("\nPlant -> disease mapping")
    print("-" * 60)
    for plant_name, disease_entries in plant_dictionary.items():
        print(f"{plant_name}: {len(disease_entries)} classes")

    # Step 2: Randomly select one valid image for each disease class.
    sample_images = get_sample_images(dataset_info, seed=args.seed)
    plant_samples = organize_samples_by_plant(sample_images)

    if not plant_samples:
        raise RuntimeError("No valid sample images were found. Check the dataset contents.")

    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving plant grids")
    print("-" * 60)

    for plant_name in sorted(plant_samples):
        samples = plant_samples[plant_name]
        fig, _ = create_grid(
            plant_name=plant_name,
            plant_samples=samples,
            rows=args.rows,
            cols=args.cols,
        )

        output_file = f"{safe_file_name(plant_name)}.png"
        output_path = os.path.join(output_dir, output_file)

        # Step 5: Save the output image.
        save_output(fig, output_path)
        print(
            f"Saved {plant_name} grid with "
            f"{min(len(samples), max_images_per_plant)} images -> {output_path}"
        )

        # Step 3 and 6: Display the grid unless disabled.
        if not args.no_display:
            plt.show(block=False)
            plt.pause(0.1)

        plt.close(fig)

    print(f"\nFinished. All plant grids are saved in: {output_dir}")


if __name__ == "__main__":
    main()
