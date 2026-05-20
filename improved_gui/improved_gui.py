from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple

from build_disease_solutions import get_solution, load_disease_solutions, normalize_key
from PySide6.QtCore import QProcess, QTimer, Qt, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QPoint, QUrl
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QFont, QPixmap, QDragEnterEvent, QDropEvent, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget, QListWidget, QMenuBar, QMenu, QAction, QListWidgetItem
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.h5"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
SUPPORTED_PLANTS_PATH = BASE_DIR / "supported_plants.txt"
EVALUATION_SCRIPT_PATH = BASE_DIR / "evaluation_model.py"
COLLEGE_LOGO_PATH = BASE_DIR / "UI_media" / "college_Logo.png"
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ... (same utility functions as old: load_class_names, count_images, format_metric, is_healthy_label, format_prediction_label, suggested_solution, load_supported_plants_data, resolve_python_executable)

# Paste all utility functions from old plant_disease_gui.py here for completeness (load_class_names to resolve_python_executable)

def load_class_names(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as handle:
        class_indices = json.load(handle)

    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    total = 0
    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "--"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "--"

    if numeric_value <= 1:
        numeric_value *= 100

    return f"{numeric_value:.2f}%"


def is_healthy_label(raw_label: str) -> bool:
    return "healthy" in raw_label.lower()


def format_prediction_label(raw_label: str) -> str:
    text = raw_label.lower()
    text = text.replace("haunglongbing", "huanglongbing")
    text = text.replace("cherry_(including_sour)", "sour_cherry")
    text = text.replace("corn_(maize)", "corn")
    text = text.replace("grape_esca_(black_measles)", "grape_esca")
    text = text.replace("grape_leaf_blight_(isariopsis_leaf_spot)", "grape_leaf_blight")
    text = text.replace("orange_huanglongbing_(citrus_greening)", "orange_huanglongbing")
    text = text.replace(",", "")
    text = text.replace("(", " ").replace(")", " ")

    tokens = [token for token in text.split("_") if token]
    deduped_tokens: List[str] = []
    for token in tokens:
        if not deduped_tokens or deduped_tokens[-1] != token:
            deduped_tokens.append(token)

    formatted = " ".join(deduped_tokens).strip()
    return formatted.title() if formatted else raw_label


def suggested_solution(raw_label: str) -> str:
    label = raw_label.lower()

    if "healthy" in label:
        return "Leaf appears healthy. Continue regular monitoring, balanced nutrition, and proper watering."
    if "virus" in label or "huanglongbing" in label:
        return "Isolate infected plants, control insect vectors, and remove severely affected leaves or plants."
    if "blight" in label:
        return "Remove infected leaves, improve airflow, avoid overhead watering, and use a recommended fungicide."
    if "rust" in label:
        return "Prune affected foliage, reduce leaf wetness, and apply a suitable fungicide if infection spreads."
    if "mildew" in label:
        return "Keep foliage dry, improve spacing, and apply sulfur or a crop-safe fungicide when required."
    if "spot" in label or "scab" in label:
        return "Remove infected leaves, sanitize tools, and apply a preventive copper-based or suitable fungicide spray."
    if "rot" in label or "mold" in label:
        return "Discard infected parts, improve drainage and ventilation, and reduce excess moisture around the plant."
    if "mite" in label:
        return "Inspect leaf undersides, isolate the plant, and use a recommended miticide or neem-based treatment."

    return "Monitor the plant closely, remove visibly infected areas, and follow crop-specific disease management practices."


def load_supported_plants_data(file_path: Path) -> dict:
    data: dict = {
        "project_title": "",
        "model_path": "",
        "source": "",
        "summary": [],
        "plants": [],
        "categories": [],
        "notes": [],
    }

    if not file_path.exists():
        return data

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return data

    current_section = "header"
    current_category_title = ""
    current_category_items: List[str] = []

    def flush_category() -> None:
        nonlocal current_category_title, current_category_items
        if current_category_title:
            categories = data["categories"]
            if isinstance(categories, list):
                categories.append({
                    "title": current_category_title,
                    "items": current_category_items[:],
                })
        current_category_title = ""
        current_category_items = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line == "Supported Plants:":
            flush_category()
            current_section = "plants"
            continue

        if line == "Disease / Health Categories by Plant:":
            flush_category()
            current_section = "categories"
            continue

        if line == "Note:":
            flush_category()
            current_section = "notes"
            continue

        if current_section == "header":
            if not data["project_title"]:
                data["project_title"] = line
            elif line.startswith("Supported Plants for Model:"):
                data["model_path"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                data["source"] = line.split(":", 1)[1].strip()
            else:
                summary = data["summary"]
                if isinstance(summary, list):
                    summary.append(line)
            continue

        if current_section == "plants":
            plants = data["plants"]
            if isinstance(plants, list):
                plant_name = line.split(". ", 1)[1] if ". " in line else line.lstrip("- ").strip()
                plants.append(plant_name)
            continue

        if current_section == "categories":
            if line.endswith(":") and not line.startswith("- "):
                flush_category()
                current_category_title = line[:-1]
            elif line.startswith("- "):
                current_category_items.append(line[2:].strip())
            continue

        if current_section == "notes":
            notes = data["notes"]
            if isinstance(notes, list):
                notes.append(line[2:].strip() if line.startswith("- ") else line)

    flush_category()
    return data


def resolve_python_executable() -> str:
    current_python = Path(sys.executable)
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"

    if current_python.exists() and current_python.name.lower() == "python.exe":
        return str(current_python)
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


class ImagePreviewLabel(QLabel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._placeholder_text = "Drag & drop or click to select leaf image"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("imagePreview")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(350)
        self.setMaximumHeight(500)
        self.setText(self._placeholder_text)
        self.setAcceptDrops(True)  # New: Drag & drop

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update_scaled_pixmap()

    def clear_preview(self) -> None:
        self._pixmap = None
        super().setPixmap(QPixmap())
        self.setText(self._placeholder_text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self) -> None:
        if not self._pixmap or self._pixmap.isNull():
            super().setPixmap(QPixmap())
            self.setText(self._placeholder_text)
            return

        scaled = self._pixmap.scaled(
            max(1, self.width() - 40),
            max(1, self.height() - 40),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)
        self.setText("")

    # New: Drag & drop support
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.parent().parent().select_image_dropped(file_path)  # Call parent's method
            event.acceptProposedAction()
        super().dropEvent(event)


class ImprovedMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.selected_image_path: Optional[Path] = None
        self.model = None
        self.tf = None
        self.np = None
        self.class_names: List[str] = []
        self.evaluation_process: Optional[QProcess] = None
        self.predictions_history: List[dict] = []  # New: History
        self.theme = 'dark'  # New: Theme state

        self.setWindowTitle("Improved Plant Leaf Disease Detection - v2")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 900)

        self._build_ui()
        self._apply_styles()
        self.reset_result_card()
        self.statusBar().showMessage("Ready - Improved GUI v2")

        QTimer.singleShot(0, self.load_runtime_information)
        self._setup_menu_bar()  # New: Menu bar

    def _setup_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        theme_menu = QMenu("Theme", self)
        dark_action = QAction("Dark", self)
        light_action = QAction("Light", self)
        dark_action.triggered.connect(lambda: self.toggle_theme('dark'))
        light_action.triggered.connect(lambda: self.toggle_theme('light'))
        dark_action.setCheckable(True)
        light_action.setCheckable(True)
        dark_action.setChecked(True)
        theme_menu.addAction(dark_action)
        theme_menu.addAction(light_action)
        menu_bar.addMenu(theme_menu)

    def toggle_theme(self, new_theme: str) -> None:
        self.theme = new_theme
        self._apply_styles()

    def _apply_styles(self) -> None:
        if self.theme == 'light':
            stylesheet = """
            QWidget { background-color: #f5f7fa; color: #1a202c; font-family: "Segoe UI"; font-size: 14px; }
            /* Light theme styles - adapt old CSS to light */
            QFrame#sidebar { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
            /* ... full light styles similar to dark but inverted colors */
            """  # Placeholder - full light styles would be defined here
        else:
            stylesheet = """
            /* Same as old dark theme, plus improvements */
            QWidget {
                background-color: #0f141b;
                color: #e8edf5;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            /* Glassmorphism cards */
            QFrame#card {
                background-color: rgba(22, 29, 38, 0.8);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(35, 45, 57, 0.5);
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            /* Gradient buttons */
            QPushButton#primaryButton {
                background: linear-gradient(135deg, #6ddc7b, #45c972);
                border-radius: 16px;
                font-weight: 700;
            }
            /* ... rest from old, enhanced */
            """  # Full stylesheet from old + glass/gradients/shadows

        self.setStyleSheet(stylesheet)

    # _build_ui, create_sidebar, create_home_page etc - enhance home_page

    def create_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Improved Plant Leaf Disease Detection v2")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Enhanced AI analysis with drag-drop, history, top preds, themes")
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.image_preview = ImagePreviewLabel()  # Enhanced with drag

        button_row = QHBoxLayout()
        self.select_button = QPushButton("📁 Select Image")
        self.select_button.setObjectName("secondaryButton")
        self.select_button.clicked.connect(self.select_image)

        self.predict_button = QPushButton("🔮 Predict")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict_disease)

        button_row.addWidget(self.select_button)
        button_row.addWidget(self.predict_button)
        button_row.addStretch()

        # Enhanced main split
        self.home_main_split = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.home_main_split.setSpacing(20)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_preview, 1)
        left_layout.addLayout(button_row)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        # Result card enhanced
        result_card, result_layout = self.create_card()
        result_title = QLabel("Detection Results")
        self.result_disease_label = QLabel("Ready")
        self.confidence_bar = QProgressBar()
        self.status_value_label = QLabel("Status")
        self.solution_label = QLabel("Ready for prediction")
        # New: Top 3 preds
        self.top_preds_layout = QVBoxLayout()
        self.top_pred_labels = [QLabel(f"#{i+1}: --") for i in range(3)]
        for lbl in self.top_pred_labels:
            lbl.setObjectName("topPred")
            self.top_preds_layout.addWidget(lbl)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_disease_label, 0, Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.confidence_bar)
        result_layout.addWidget(self.status_value_label)
        result_layout.addLayout(self.top_preds_layout)
        result_layout.addWidget(self.solution_label)
        result_layout.addStretch()

        # New export btn
        self.export_btn = QPushButton("💾 Export Result")
        self.export_btn.clicked.connect(self.export_result)
        self.export_btn.setObjectName("secondaryButton")
        self.export_btn.setEnabled(False)
        result_layout.addWidget(self.export_btn)

        # How to
        how_to_card, _ = self.create_card()
        # ...

        # New history section
        history_card, history_layout = self.create_card()
        history_title = QLabel("Recent Predictions")
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        clear_history_btn = QPushButton("Clear")
        clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(history_title)
        history_layout.addWidget(self.history_list)
        history_layout.addWidget(clear_history_btn)

        right_layout.addWidget(result_card)
        right_layout.addWidget(history_card)
        right_layout.addStretch()

        self.home_main_split.addLayout(left_layout, 2)
        self.home_main_split.addLayout(right_layout, 1)

        layout.addLayout(self.home_main_split, 1)
        return page

    # Other pages same as old for brevity

    def create_card(self) -> Tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        return card, layout

    # ... (copy other create_page methods from old code for completeness)

    # Assume they are copied here...

    def select_image_dropped(self, file_path: str) -> None:
        pixmap = QPixmap(file_path)
        self.selected_image_path = Path(file_path)
        self.image_preview.set_preview_pixmap(pixmap)
        self.predict_button.setEnabled(True)
        self.reset_result_card()

    # Enhanced predict to return top3
    def run_prediction(self, image_path: Path) -> dict:
        # Same as old
        # ...
        predictions = self.model.predict(image_array, verbose=0)[0]
        predicted_index = int(self.np.argmax(predictions))
        confidence = float(predictions[predicted_index] * 100)

        # New: Top 3
        top_indices = self.np.argsort(predictions)[-3:][::-1]
        top3 = [(self.class_names[i], float(predictions[i]*100)) for i in top_indices]

        # ... rest same
        return {
            "raw_label": raw_label,
            "display_label": display_label,
            "confidence": confidence,
            "status": "Healthy" if is_healthy_label(raw_label) else "Diseased",
            "solution": solution_text,
            "top3": top3  # New
        }

    def show_prediction_result(self, result: dict) -> None:
        # Same + new
        # Top preds
        for i, (label, conf) in enumerate(result["top3"]):
            self.top_pred_labels[i].setText(f"#{i+1}: {format_prediction_label(label)} ({conf:.1f}%)")

        # Animate show
        anim_group = QParallelAnimationGroup()
        result_anim = QPropertyAnimation(self.result_disease_label, b"windowOpacity")
        result_anim.setDuration(500)
        result_anim.setStartValue(0.0)
        result_anim.setEndValue(1.0)
        result_anim.setEasingCurve(QEasingCurve.Type.OutBounce)
        anim_group.addAnimation(result_anim)
        anim_group.start()

        # History
        history_item = f"{result['display_label']} - {result['confidence']:.1f}% ({result['status']})"
        self.predictions_history.append({
            "label": result["display_label"],
            "conf": result["confidence"],
            "status": result["status"],
            "time": "now"
        })
        if len(self.predictions_history) > 5:
            self.predictions_history.pop(0)
        self.history_list.clear()
        for h in self.predictions_history[-5:]:
            item = QListWidgetItem(f"{h['label']} ({h['conf']:.1f}%)")
            self.history_list.addItem(item)

        self.export_btn.setEnabled(True)

    def export_result(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", "prediction.png", "PNG (*.png)")
        if file_path:
            pixmap = QPixmap.grabWidget(self.result_card)
            pixmap.save(file_path)

    def clear_history(self) -> None:
        self.history_list.clear()
        self.predictions_history.clear()

    # Other methods same as old: load_runtime_information, etc.

    # Assume all other methods copied from old for full functionality

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Leaf Image", str(BASE_DIR), "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if file_path:
            self.selected_image_path = Path(file_path)
            pixmap = QPixmap(file_path)
            self.image_preview.set_preview_pixmap(pixmap)
            self.predict_button.setEnabled(True)
            self.reset_result_card()
        self.export_btn.setEnabled(False)

    # ... rest of methods identical to old plant_disease_gui.py (predict_disease, load_model_if_needed, etc.)

    # Paste them here to make complete file


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = ImprovedMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

