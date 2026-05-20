import argparse
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model, mixed_precision
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = "final_dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LAYERS = 30
SEED = 42
BEST_MODEL_PATH = "best_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
PLOT_PATH = "training_curves.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 plant disease classifier.")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Prepared dataset root containing train/val/test.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training and evaluation.")
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS, help="Epochs for frozen-base training.")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Epochs for fine-tuning.")
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=FINE_TUNE_LAYERS,
        help="How many top MobileNetV2 layers to unfreeze for fine-tuning.",
    )
    parser.add_argument("--best-model-path", default=BEST_MODEL_PATH, help="Path to save the best model.")
    parser.add_argument("--class-indices-path", default=CLASS_INDICES_PATH, help="Path to save class_indices.json.")
    parser.add_argument("--plot-path", default=PLOT_PATH, help="Path to save the training curves plot.")
    parser.add_argument(
        "--enable-xla",
        action="store_true",
        help="Enable XLA JIT compilation. Keep disabled unless CUDA toolkit/ptxas is configured correctly.",
    )
    return parser.parse_args()


def resolve_project_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_hardware(enable_xla=False):
    print("\n========== TensorFlow Runtime ==========")
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    use_mixed_precision = False

    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for index, gpu in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"Could not enable memory growth for {gpu.name}: {exc}")

            details = {}
            try:
                details = tf.config.experimental.get_device_details(gpu)
            except Exception:
                details = {}

            device_name = details.get("device_name", gpu.name)
            compute_capability = details.get("compute_capability")
            print(f"GPU {index}: {device_name}")
            if compute_capability:
                print(f"  Compute capability: {compute_capability[0]}.{compute_capability[1]}")
                if compute_capability[0] >= 7:
                    use_mixed_precision = True
            else:
                print("  Compute capability: unavailable")

        if use_mixed_precision:
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision: enabled")
        else:
            print("Mixed precision: disabled")

        if enable_xla:
            try:
                tf.config.optimizer.set_jit(True)
                print("XLA JIT: enabled")
            except Exception:
                print("XLA JIT: unavailable")
        else:
            try:
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass
            print("XLA JIT: disabled")
    else:
        print("No GPU detected. Training will run on CPU.")
        print("Mixed precision: disabled")
        print("XLA JIT: disabled")

    print(f"Global policy: {mixed_precision.global_policy()}")
    print("========================================\n")


def validate_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")

    return train_dir, val_dir, test_dir


def count_images(directory):
    total = 0
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    for root, _, files in os.walk(directory):
        total += sum(file_name.lower().endswith(valid_extensions) for file_name in files)
    return total


def create_generators(train_dir, val_dir, test_dir, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
        interpolation="bilinear",
    )
    val_generator = eval_datagen.flow_from_directory(
        str(val_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        interpolation="bilinear",
    )
    test_generator = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        interpolation="bilinear",
    )

    print("\n========== Dataset Summary ==========")
    print(f"Train images: {count_images(train_dir)}")
    print(f"Val images:   {count_images(val_dir)}")
    print(f"Test images:  {count_images(test_dir)}")
    print(f"Classes:      {train_generator.num_classes}")
    print("=====================================\n")

    return train_generator, val_generator, test_generator


def save_class_indices(class_indices, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(class_indices, handle, indent=2)
    print(f"Saved class indices to: {output_path}")


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs, outputs, name="plant_disease_mobilenetv2")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def create_callbacks(best_model_path):
    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def fit_with_fallback(model, train_generator, val_generator, epochs, initial_epoch, callbacks):
    fit_kwargs = {
        "x": train_generator,
        "validation_data": val_generator,
        "epochs": epochs,
        "initial_epoch": initial_epoch,
        "callbacks": callbacks,
        "verbose": 1,
    }

    # Windows + Keras ImageDataGenerator can fail with multiprocessing because
    # the spawned workers must pickle generator state that contains thread locks.
    if os.name == "nt":
        thread_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        print(f"Using Windows-safe threaded data loading with workers={thread_workers}.")
        return model.fit(
            **fit_kwargs,
            workers=thread_workers,
            use_multiprocessing=False,
            max_queue_size=32,
        )

    try:
        return model.fit(
            **fit_kwargs,
            workers=max(1, (os.cpu_count() or 2) - 1),
            use_multiprocessing=True,
            max_queue_size=32,
        )
    except TypeError:
        print("Keras fit() multiprocessing arguments are unavailable in this version. Falling back.")
        return model.fit(**fit_kwargs)


def unfreeze_top_layers(base_model, fine_tune_layers):
    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-fine_tune_layers:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    trainable_layers = sum(int(layer.trainable) for layer in base_model.layers)
    print(f"Fine-tuning with {trainable_layers} trainable base-model layers.")


def merge_histories(histories):
    merged = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def plot_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, label="Val Accuracy", linewidth=2)
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {output_path}")


def evaluate_model(model, test_generator, class_names):
    print("\n========== Test Evaluation ==========")
    test_generator.reset()
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes

    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=====================================\n")


def main():
    args = parse_args()
    set_seed(SEED)

    dataset_path = resolve_project_path(args.dataset_path)
    best_model_path = resolve_project_path(args.best_model_path)
    class_indices_path = resolve_project_path(args.class_indices_path)
    plot_path = resolve_project_path(args.plot_path)

    train_dir, val_dir, test_dir = validate_dataset(dataset_path)
    configure_hardware(enable_xla=args.enable_xla)

    train_generator, val_generator, test_generator = create_generators(
        train_dir,
        val_dir,
        test_dir,
        args.batch_size,
    )

    class_names = [name for name, _ in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
    save_class_indices(train_generator.class_indices, class_indices_path)

    model, base_model = build_model(train_generator.num_classes)
    compile_model(model, learning_rate=1e-3)

    print("Starting initial training...")
    initial_history = fit_with_fallback(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=args.initial_epochs,
        initial_epoch=0,
        callbacks=create_callbacks(best_model_path),
    )

    completed_initial_epochs = len(initial_history.history.get("loss", []))
    total_epochs = completed_initial_epochs + args.fine_tune_epochs

    print("Starting fine-tuning...")
    unfreeze_top_layers(base_model, args.fine_tune_layers)
    compile_model(model, learning_rate=1e-5)

    fine_tune_history = fit_with_fallback(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=total_epochs,
        initial_epoch=completed_initial_epochs,
        callbacks=create_callbacks(best_model_path),
    )

    full_history = merge_histories([initial_history, fine_tune_history])
    plot_history(full_history, plot_path)

    best_model = tf.keras.models.load_model(str(best_model_path), compile=False)
    compile_model(best_model, learning_rate=1e-5)
    evaluate_model(best_model, test_generator, class_names)

    print("Artifacts saved:")
    print(f"  best model:      {best_model_path}")
    print(f"  class indices:   {class_indices_path}")
    print(f"  training curves: {plot_path}")


if __name__ == "__main__":
    main()
