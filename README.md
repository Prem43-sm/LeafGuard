# LeafGuard
<img width="720" height="720" alt="image" src="https://github.com/user-attachments/assets/22fbcc84-3512-4aca-9ab3-8b9b0df49b42" />

Plant leaf disease detection system built with TensorFlow and PySide6.

## What is included

- Training pipeline for a MobileNetV2-based plant disease classifier.
- Dataset merge, cleanup, resize, and split utilities.
- PySide6 desktop GUI for loading leaf images and viewing disease predictions.
- Improved GUI prototype with drag-and-drop, top predictions, history, and export support.
- Evaluation scripts for metrics, reports, confusion matrices, and prediction visualizations.
- Disease solution metadata and supported plant references.

## What is intentionally excluded

Large or generated files are ignored and not committed:

- `final_dataset/`
- `test/`
- `dataset sample images/`
- `UI_media/`
- `improved_gui/UI_media/`
- `*.h5` model weights
- `best_model_*/` evaluation output folders
- generated images such as `training_curves.png`
- `.venv/` and `__pycache__/`

Add those files locally when training, evaluating, or running the full GUI with a trained model.

## Setup

Use Python 3.10 on Windows for the pinned TensorFlow version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Common commands

Train a model:

```powershell
python train_plant_model.py
```

Run the main GUI after adding `best_model.h5`:

```powershell
python plant_disease_gui.py
```

Run the improved GUI after adding its required local model/assets:

```powershell
python improved_gui\improved_gui.py
```

Evaluate the trained model:

```powershell
python test_model.py
```
