import argparse
import hashlib
import multiprocessing
import os
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET1_PATH = "New Plant Diseases Dataset(Augmented)"
DEFAULT_DATASET2_PATH = "PlantVillage"
DEFAULT_OUTPUT_DIR = "final_dataset"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_JPEG_QUALITY = 90
DEFAULT_MAX_PER_CLASS = 2000
DEFAULT_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

WORKER_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
WORKER_JPEG_QUALITY = DEFAULT_JPEG_QUALITY
WORKER_USE_GPU = False


def standardize_class_name(class_name):
    class_name = class_name.strip()
    class_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", class_name)
    class_name = class_name.replace("___", "_")
    class_name = class_name.replace("-", "_").replace(" ", "_")
    class_name = class_name.lower()
    class_name = re.sub(r"_+", "_", class_name)
    return class_name.strip("_")


def iter_image_files(root_dir):
    stack = [os.path.abspath(root_dir)]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        extension = os.path.splitext(entry.name)[1].lower()
                        if extension in IMAGE_EXTENSIONS:
                            yield entry.path
        except OSError:
            continue


def compute_md5(file_path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    try:
        with open(file_path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def load_and_clean(dataset_paths, max_per_class=DEFAULT_MAX_PER_CLASS, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    class_candidates = defaultdict(list)
    total_scanned = 0

    for dataset_path in dataset_paths:
        dataset_path = os.path.abspath(dataset_path)
        if not os.path.isdir(dataset_path):
            print(f"Skipping missing dataset: {dataset_path}")
            continue

        dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
        for image_path in tqdm(iter_image_files(dataset_path), desc=f"Scanning {dataset_name}", unit="img"):
            parent_dir = os.path.basename(os.path.dirname(image_path))
            class_name = standardize_class_name(parent_dir)
            if not class_name:
                continue
            class_candidates[class_name].append(image_path)
            total_scanned += 1

    seen_hashes = set()
    duplicate_count = 0
    hash_failures = 0
    cleaned = {}

    for class_name in sorted(class_candidates):
        candidates = class_candidates[class_name]
        rng.shuffle(candidates)

        selected = []
        for image_path in candidates:
            if len(selected) >= max_per_class:
                break

            file_hash = compute_md5(image_path)
            if file_hash is None:
                hash_failures += 1
                continue
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(file_hash)
            selected.append((image_path, file_hash))

        if selected:
            cleaned[class_name] = selected

    class_counts = {class_name: len(records) for class_name, records in cleaned.items()}

    print(f"Total scanned images: {total_scanned}")
    print(f"Total classes: {len(class_counts)}")
    print("Images per class:")
    for class_name in sorted(class_counts):
        print(f"  {class_name}: {class_counts[class_name]}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Hash/read failures skipped: {hash_failures}")
    print(f"Total unique selected images: {sum(class_counts.values())}")

    return cleaned


def init_worker(image_size, jpeg_quality, use_gpu):
    global WORKER_IMAGE_SIZE, WORKER_JPEG_QUALITY, WORKER_USE_GPU

    WORKER_IMAGE_SIZE = int(image_size)
    WORKER_JPEG_QUALITY = int(jpeg_quality)
    WORKER_USE_GPU = bool(use_gpu and torch is not None and torch.cuda.is_available())

    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    if torch is not None:
        torch.set_num_threads(1)


def resize_with_cpu(rgb_image):
    interpolation = (
        cv2.INTER_AREA
        if rgb_image.shape[0] >= WORKER_IMAGE_SIZE and rgb_image.shape[1] >= WORKER_IMAGE_SIZE
        else cv2.INTER_LINEAR
    )
    return cv2.resize(rgb_image, (WORKER_IMAGE_SIZE, WORKER_IMAGE_SIZE), interpolation=interpolation)


def resize_with_gpu(rgb_image):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_image))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    tensor = tensor.to("cuda", non_blocking=True)
    resized = F.interpolate(
        tensor,
        size=(WORKER_IMAGE_SIZE, WORKER_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.mul_(255.0).clamp_(0, 255).byte()
    return resized.squeeze(0).permute(1, 2, 0).cpu().numpy()


def process_image(task):
    source_path, output_path = task

    try:
        buffer = np.fromfile(source_path, dtype=np.uint8)
        if buffer.size == 0:
            return False, output_path, False, f"Empty file: {source_path}"

        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return False, output_path, False, f"Corrupted image: {source_path}"

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        used_gpu = False

        if WORKER_USE_GPU:
            try:
                rgb_image = resize_with_gpu(rgb_image)
                used_gpu = True
            except Exception:
                rgb_image = resize_with_cpu(rgb_image)
        else:
            rgb_image = resize_with_cpu(rgb_image)

        output_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success, encoded = cv2.imencode(
            ".jpg",
            output_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, WORKER_JPEG_QUALITY],
        )
        if not success:
            return False, output_path, used_gpu, f"Encode failed: {source_path}"

        encoded.tofile(output_path)
        return True, output_path, used_gpu, ""
    except Exception as exc:
        return False, output_path, False, f"{source_path}: {exc}"


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def split_one_class(paths, seed):
    if not paths:
        return [], [], []

    if len(paths) == 1:
        return paths, [], []

    if len(paths) == 2:
        return [paths[0]], [paths[1]], []

    train_paths, remainder = train_test_split(
        paths,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )

    if len(remainder) == 1:
        return train_paths, remainder, []

    val_paths, test_paths = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )
    return train_paths, val_paths, test_paths


def split_dataset(processed_by_class, output_dir, seed=DEFAULT_SEED):
    split_totals = {"train": 0, "val": 0, "test": 0}

    for split_name in split_totals:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    for class_name in sorted(processed_by_class):
        class_paths = list(processed_by_class[class_name])
        train_paths, val_paths, test_paths = split_one_class(class_paths, seed)

        for split_name, split_paths in (
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ):
            destination_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(destination_dir, exist_ok=True)
            for source_path in split_paths:
                destination_path = os.path.join(destination_dir, os.path.basename(source_path))
                shutil.move(source_path, destination_path)
                split_totals[split_name] += 1

    return split_totals


def parse_args():
    parser = argparse.ArgumentParser(description="Merge, preprocess, deduplicate, and split plant disease datasets.")
    parser.add_argument("--dataset1", default=DEFAULT_DATASET1_PATH, help="Path to the first dataset.")
    parser.add_argument("--dataset2", default=DEFAULT_DATASET2_PATH, help="Path to the second dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for the merged dataset.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Target square image size.")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality for saved images.")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS, help="Maximum images to keep per class.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Number of worker processes.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for dispatching image tasks.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU resizing with torch if CUDA is available.")
    return parser.parse_args()


def resolve_project_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_paths = [str(resolve_project_path(args.dataset1)), str(resolve_project_path(args.dataset2))]
    output_dir = str(resolve_project_path(args.output_dir))
    staging_dir = os.path.join(output_dir, "_staging")
    num_workers = max(1, int(args.workers))
    batch_size = max(1, int(args.batch_size))
    gpu_available = bool(torch is not None and torch.cuda.is_available())
    use_gpu = bool(args.use_gpu and gpu_available)

    print(f"CPU workers: {num_workers}")
    print(f"GPU available: {gpu_available}")
    print(f"GPU being used: {use_gpu}")

    cleaned_records = load_and_clean(
        dataset_paths=dataset_paths,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    if not cleaned_records:
        print("No valid images found. Exiting.")
        return

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(staging_dir, exist_ok=True)

    processed_by_class = defaultdict(list)
    processed_total = 0
    failed_total = 0
    gpu_processed = 0
    sample_errors = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(args.image_size, args.jpeg_quality, use_gpu),
    ) as executor:
        for class_name in sorted(cleaned_records):
            tasks = [
                (
                    source_path,
                    os.path.join(staging_dir, class_name, f"{file_hash}.jpg"),
                )
                for source_path, file_hash in cleaned_records[class_name]
            ]

            class_chunk_size = max(1, min(32, len(tasks) // max(1, num_workers * 4) or 1))

            with tqdm(total=len(tasks), desc=f"Processing {class_name}", unit="img") as progress_bar:
                for batch in iter_batches(tasks, batch_size):
                    for success, output_path, used_gpu_for_image, error_message in executor.map(
                        process_image,
                        batch,
                        chunksize=class_chunk_size,
                    ):
                        progress_bar.update(1)
                        if success:
                            processed_by_class[class_name].append(output_path)
                            processed_total += 1
                            if used_gpu_for_image:
                                gpu_processed += 1
                        else:
                            failed_total += 1
                            if len(sample_errors) < 20:
                                sample_errors.append(error_message)

    split_totals = split_dataset(processed_by_class, output_dir, seed=args.seed)
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Total processed images: {processed_total}")
    print(f"Failed/corrupted images skipped: {failed_total}")
    print(f"GPU-resized images: {gpu_processed}")
    print(f"Train images: {split_totals['train']}")
    print(f"Val images: {split_totals['val']}")
    print(f"Test images: {split_totals['test']}")

    if sample_errors:
        print("Sample skipped files:")
        for error_message in sample_errors:
            print(f"  {error_message}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
