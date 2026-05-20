from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from build_disease_solutions import get_solution, load_disease_solutions, normalize_key
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QFont, QPixmap
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
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.h5"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
DATASET_DIR = BASE_DIR / "final_dataset"
METRICS_PATH = BASE_DIR / "model_metrics.json"
SUPPORTED_PLANTS_PATH = BASE_DIR / "supported_plants.txt"
EVALUATION_SCRIPT_PATH = BASE_DIR / "evaluation_model.py"
COLLEGE_LOGO_PATH = BASE_DIR / "UI_media" / "college_Logo.png"
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_names(file_path: Path) -> list[str]:
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
    deduped_tokens: list[str] = []
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


def load_supported_plants_data(file_path: Path) -> dict[str, object]:
    data: dict[str, object] = {
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
    current_category_items: list[str] = []

    def flush_category() -> None:
        nonlocal current_category_title, current_category_items
        if current_category_title:
            categories = data["categories"]
            if isinstance(categories, list):
                categories.append(
                    {
                        "title": current_category_title,
                        "items": current_category_items[:],
                    }
                )
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
        self._placeholder_text = "Select a leaf image"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("imagePreview")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(350)
        self.setMaximumHeight(500)
        self.setText(self._placeholder_text)

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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.selected_image_path: Optional[Path] = None
        self.model = None
        self.tf = None
        self.np = None
        self.class_names: list[str] = []
        self.evaluation_process: Optional[QProcess] = None

        self.setWindowTitle("Plant Leaf Disease Detection System")
        self.setMinimumSize(1000, 680)
        self.resize(1360, 860)

        self._build_ui()
        self._apply_styles()
        self.reset_result_card()
        self.statusBar().showMessage("Ready")

        QTimer.singleShot(0, self.load_runtime_information)

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = self.create_sidebar()
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.home_page = self.create_home_page()
        self.supported_plants_page = self.create_supported_plants_page()
        self.project_details_page = self.create_project_details_page()
        self.technical_page = self.create_technical_model_page()
        self.about_page = self.create_about_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.supported_plants_page)
        self.stack.addWidget(self.project_details_page)
        self.stack.addWidget(self.technical_page)
        self.stack.addWidget(self.about_page)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)

        self.switch_page(0)

    def create_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMaximumWidth(180)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(10)

        brand_title = QLabel("Plant Leaf\nAI Detect")
        brand_title.setObjectName("brandTitle")

        brand_subtitle = QLabel("Smart disease analysis")
        brand_subtitle.setObjectName("brandSubtitle")
        brand_subtitle.setWordWrap(True)

        layout.addWidget(brand_title)
        layout.addWidget(brand_subtitle)

        nav_wrapper = QVBoxLayout()
        nav_wrapper.setSpacing(10)
        nav_wrapper.setContentsMargins(0, 12, 0, 0)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[QPushButton] = []

        for index, label in enumerate(["Home", "Supported Plants", "Project Details", "Technical & Model", "About"]):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setObjectName("sidebarButton")
            button.clicked.connect(lambda checked=False, page_index=index: self.switch_page(page_index))
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            nav_wrapper.addWidget(button)

        layout.addLayout(nav_wrapper)
        layout.addStretch(1)

        return sidebar

    def create_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Plant Leaf Disease Detection System")
        title.setObjectName("pageTitle")

        subtitle = QLabel("AI-based identification of plant diseases from leaf images")
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.image_preview = ImagePreviewLabel()

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.select_button = QPushButton("Select Image")
        self.select_button.setObjectName("secondaryButton")
        self.select_button.clicked.connect(self.select_image)

        self.predict_button = QPushButton("Predict")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict_disease)

        button_row.addWidget(self.select_button)
        button_row.addWidget(self.predict_button)
        button_row.addStretch(1)

        result_card, result_layout = self.create_card()

        result_title = QLabel("Detection Result")
        result_title.setObjectName("cardTitle")

        self.result_disease_label = QLabel("No prediction yet")
        self.result_disease_label.setObjectName("resultTitle")
        self.result_disease_label.setWordWrap(True)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("--")
        self.confidence_bar.setTextVisible(True)

        self.status_value_label = QLabel("Status")
        self.status_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_value_label.setObjectName("statusBadge")

        self.solution_label = QLabel("Solution will appear here")
        self.solution_label.setWordWrap(True)
        self.solution_label.setObjectName("mutedLabel")

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_disease_label)
        result_layout.addWidget(self.confidence_bar)
        result_layout.addWidget(self.status_value_label)
        result_layout.addWidget(self.solution_label)
        result_layout.addStretch(1)

        how_to_card, how_to_layout = self.create_card()
        how_to_title = QLabel("How to Use")
        how_to_title.setObjectName("cardTitle")

        how_to_steps = QLabel("1. Select a leaf image\n2. Click Predict\n3. View result")
        how_to_steps.setObjectName("mutedLabel")
        how_to_steps.setWordWrap(True)

        how_to_layout.addWidget(how_to_title)
        how_to_layout.addWidget(how_to_steps)
        how_to_layout.addStretch(1)

        self.home_main_split = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.home_main_split.setSpacing(16)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(14)
        left_layout.addWidget(self.image_preview, 1)
        left_layout.addLayout(button_row)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)
        right_layout.addWidget(result_card)
        right_layout.addWidget(how_to_card)
        right_layout.addStretch(1)

        self.home_main_split.addLayout(left_layout, 2)
        self.home_main_split.addLayout(right_layout, 1)

        layout.addLayout(self.home_main_split, 1)
        return page

    def create_supported_plants_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("Supported Plants")
        title.setObjectName("pageTitle")

        subtitle = QLabel("Plant types and disease categories supported by the trained model")
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(16)

        supported_data = load_supported_plants_data(SUPPORTED_PLANTS_PATH)

        if not any(supported_data.values()):
            empty_card, empty_layout = self.create_card()

            empty_title = QLabel("Supported Plants File Not Found")
            empty_title.setObjectName("cardTitle")

            empty_body = QLabel("The file supported_plants.txt could not be loaded from the project folder.")
            empty_body.setWordWrap(True)
            empty_body.setObjectName("bodyText")

            empty_layout.addWidget(empty_title)
            empty_layout.addWidget(empty_body)
            scroll_layout.addWidget(empty_card)
        else:
            summary_card, summary_layout = self.create_card()

            summary_title = QLabel("Model Summary")
            summary_title.setObjectName("cardTitle")

            project_label = QLabel(str(supported_data.get("project_title", "Plant Leaf Disease Detection System")))
            project_label.setObjectName("resultTitle")
            project_label.setWordWrap(True)

            model_label = QLabel(f"Model: {supported_data.get('model_path', MODEL_PATH)}")
            model_label.setObjectName("bodyText")
            model_label.setWordWrap(True)

            source_label = QLabel(f"Source: {supported_data.get('source', 'supported_plants.txt')}")
            source_label.setObjectName("mutedLabel")
            source_label.setWordWrap(True)

            summary_layout.addWidget(summary_title)
            summary_layout.addWidget(project_label)
            summary_layout.addWidget(model_label)
            summary_layout.addWidget(source_label)

            for summary_line in supported_data.get("summary", []):
                summary_item = QLabel(str(summary_line))
                summary_item.setObjectName("bodyText")
                summary_item.setWordWrap(True)
                summary_layout.addWidget(summary_item)

            scroll_layout.addWidget(summary_card)

            plants = supported_data.get("plants", [])
            if isinstance(plants, list) and plants:
                plants_card, plants_layout = self.create_card()

                plants_title = QLabel("Supported Plant Types")
                plants_title.setObjectName("cardTitle")

                plants_body = QLabel("\n".join(f"- {plant}" for plant in plants))
                plants_body.setObjectName("bodyText")
                plants_body.setWordWrap(True)

                plants_layout.addWidget(plants_title)
                plants_layout.addWidget(plants_body)
                scroll_layout.addWidget(plants_card)

            categories = supported_data.get("categories", [])
            if isinstance(categories, list):
                categories_subtitle = QLabel("Disease / Health Categories by Plant:")
                categories_subtitle.setObjectName("cardTitle")
                scroll_layout.addWidget(categories_subtitle)

                for category in categories:
                    if not isinstance(category, dict):
                        continue

                    category_card, category_layout = self.create_card()

                    category_title = QLabel(str(category.get("title", "Plant")))
                    category_title.setObjectName("cardTitle")

                    items = category.get("items", [])
                    category_body = QLabel(
                        "\n".join(f"- {item}" for item in items) if isinstance(items, list) else ""
                    )
                    category_body.setObjectName("bodyText")
                    category_body.setWordWrap(True)

                    category_layout.addWidget(category_title)
                    category_layout.addWidget(category_body)
                    scroll_layout.addWidget(category_card)

            notes = supported_data.get("notes", [])
            if isinstance(notes, list) and notes:
                note_card, note_layout = self.create_card()

                note_title = QLabel("Note")
                note_title.setObjectName("cardTitle")

                note_body = QLabel("\n".join(f"- {note}" for note in notes))
                note_body.setObjectName("mutedLabel")
                note_body.setWordWrap(True)

                note_layout.addWidget(note_title)
                note_layout.addWidget(note_body)
                scroll_layout.addWidget(note_card)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        return page

    def create_project_details_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("Project Details")
        title.setObjectName("pageTitle")

        subtitle = QLabel("Overview of the plant disease detection project, objectives, and practical impact")
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(16)

        sections = [
            (
                "Introduction",
                "Plant diseases reduce crop quality and productivity. This system uses artificial intelligence to analyze leaf images and identify possible diseases quickly, supporting faster diagnosis and timely treatment.",
            ),
            (
                "Objective",
                "To build an intelligent desktop application that detects plant leaf diseases from images, displays clear prediction results, and helps users understand the health condition of the plant.",
            ),
            (
                "Problem Statement",
                "Traditional plant disease diagnosis can be slow, expensive, and dependent on expert availability. Farmers and students need an accessible system that can provide instant disease identification from leaf images.",
            ),
            (
                "Features",
                "- Upload leaf images from the local system\n"
                "- AI-based disease prediction using a trained deep learning model\n"
                "- Clear result display with confidence score and status\n"
                "- Suggested solution for quick response and plant care\n"
                "- Modern multi-page interface with responsive layout",
            ),
            (
                "Applications",
                "- Crop health monitoring\n"
                "- Academic and research demonstrations\n"
                "- Farmer awareness and decision support\n"
                "- Smart agriculture and precision farming systems",
            ),
        ]

        for section_title, section_body in sections:
            card, card_layout = self.create_card()

            heading = QLabel(section_title)
            heading.setObjectName("cardTitle")

            body = QLabel(section_body)
            body.setWordWrap(True)
            body.setObjectName("bodyText")

            card_layout.addWidget(heading)
            card_layout.addWidget(body)
            scroll_layout.addWidget(card)

        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        return page

    def create_technical_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(16)

        title = QLabel("Technical & Model")
        title.setObjectName("pageTitle")

        subtitle = QLabel("Implementation stack, model details, dataset summary, and evaluation placeholders")
        subtitle.setObjectName("pageSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(16)

        technologies_card, technologies_layout = self.create_card()
        technologies_title = QLabel("Technologies Used")
        technologies_title.setObjectName("cardTitle")

        technologies_text = QLabel("Python\nTensorFlow\nOpenCV\nPySide6")
        technologies_text.setObjectName("bodyText")

        technologies_layout.addWidget(technologies_title)
        technologies_layout.addWidget(technologies_text)
        scroll_layout.addWidget(technologies_card)

        model_card, model_layout = self.create_card()
        model_title = QLabel("Model")
        model_title.setObjectName("cardTitle")

        model_text = QLabel("CNN / Transfer Learning\nMobileNetV2-based classifier trained on plant leaf disease categories")
        model_text.setWordWrap(True)
        model_text.setObjectName("bodyText")

        model_layout.addWidget(model_title)
        model_layout.addWidget(model_text)
        scroll_layout.addWidget(model_card)

        dataset_card, dataset_layout = self.create_card()
        dataset_title = QLabel("Dataset Info")
        dataset_title.setObjectName("cardTitle")

        self.class_count_label = QLabel("Number of Classes: Loading...")
        self.class_count_label.setObjectName("bodyText")

        self.image_count_label = QLabel("Number of Images: Loading...")
        self.image_count_label.setObjectName("bodyText")

        self.dataset_split_label = QLabel("Split Details: Loading...")
        self.dataset_split_label.setObjectName("bodyText")
        self.dataset_split_label.setWordWrap(True)

        dataset_layout.addWidget(dataset_title)
        dataset_layout.addWidget(self.class_count_label)
        dataset_layout.addWidget(self.image_count_label)
        dataset_layout.addWidget(self.dataset_split_label)
        scroll_layout.addWidget(dataset_card)

        performance_card, performance_layout = self.create_card()
        performance_title = QLabel("Performance")
        performance_title.setObjectName("cardTitle")

        self.accuracy_label = QLabel()
        self.precision_label = QLabel()
        self.recall_label = QLabel()
        self.f1_score_label = QLabel()

        self.accuracy_label.setObjectName("metricText")
        self.precision_label.setObjectName("metricText")
        self.recall_label.setObjectName("metricText")
        self.f1_score_label.setObjectName("metricText")

        self.accuracy_label.setText("Accuracy: --")
        self.precision_label.setText("Precision: --")
        self.recall_label.setText("Recall: --")
        self.f1_score_label.setText("F1 Score: --")

        self.evaluate_model_button = QPushButton("Evaluate Model")
        self.evaluate_model_button.setObjectName("secondaryButton")
        self.evaluate_model_button.setMinimumWidth(190)
        self.evaluate_model_button.setMaximumWidth(220)
        self.evaluate_model_button.clicked.connect(self.run_model_evaluation)

        performance_note = QLabel("These values can be updated dynamically after model evaluation.")
        performance_note.setObjectName("mutedLabel")
        performance_note.setWordWrap(True)
        self.performance_note_label = performance_note

        performance_button_row = QHBoxLayout()
        performance_button_row.setSpacing(12)
        performance_button_row.addWidget(self.evaluate_model_button)
        performance_button_row.addStretch(1)

        performance_layout.addWidget(performance_title)
        performance_layout.addSpacing(4)
        performance_layout.addWidget(self.accuracy_label)
        performance_layout.addWidget(self.precision_label)
        performance_layout.addWidget(self.recall_label)
        performance_layout.addWidget(self.f1_score_label)
        performance_layout.addSpacing(8)
        performance_layout.addLayout(performance_button_row)
        performance_layout.addSpacing(4)
        performance_layout.addWidget(performance_note)
        scroll_layout.addWidget(performance_card)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        return page

    def create_about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 24, 20, 24)
        scroll_layout.setSpacing(0)

        scroll_layout.addStretch(1)

        self.about_content_layout = QVBoxLayout()
        self.about_content_layout.setSpacing(12)
        self.about_content_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        self.about_panel = QFrame()
        self.about_panel.setObjectName("aboutPanel")
        self.about_panel.setMaximumWidth(860)

        panel_layout = QVBoxLayout(self.about_panel)
        panel_layout.setContentsMargins(30, 20, 30, 24)
        panel_layout.setSpacing(12)
        panel_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        college_name = QLabel("Govt. E. Raghvendra Rao P.G. Science College\nSarkanda Bilaspur")
        college_name.setObjectName("aboutCollegeName")
        college_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        college_name.setWordWrap(True)

        self.college_logo_label = QLabel("College Logo")
        self.college_logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.college_logo_label.setObjectName("aboutLogo")
        self.college_logo_label.setMinimumSize(220, 220)
        self.college_logo_label.setMaximumSize(300, 300)

        project_name = QLabel("Plant Leaf Disease Detection System")
        project_name.setObjectName("aboutProjectName")
        project_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        project_name.setWordWrap(True)

        project_subtitle = QLabel("AI-based identification of plant diseases from leaf images")
        project_subtitle.setObjectName("aboutProjectSubtitle")
        project_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        project_subtitle.setWordWrap(True)

        developer_heading = QLabel("Developed By:")
        developer_heading.setObjectName("aboutSectionHeading")
        developer_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        developer_text = QLabel("Deepika Sahu\nInformation Technology Department")
        developer_text.setObjectName("aboutInfoText")
        developer_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        developer_text.setWordWrap(True)

        guide_heading = QLabel("Guided By:")
        guide_heading.setObjectName("aboutSectionHeading")
        guide_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        guide_text = QLabel("Dr. Sumati Pathak")
        guide_text.setObjectName("aboutInfoText")
        guide_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        guide_text.setWordWrap(True)

        hod_heading = QLabel("HOD:")
        hod_heading.setObjectName("aboutSectionHeading")
        hod_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hod_text = QLabel("Dr. Kajal Kiran Gulhare")
        hod_text.setObjectName("aboutInfoText")
        hod_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hod_text.setWordWrap(True)

        department_text = QLabel("Department of Information Technology")
        department_text.setObjectName("aboutFooterText")
        department_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        department_text.setWordWrap(True)

        panel_layout.addWidget(college_name)
        panel_layout.addWidget(self.college_logo_label, 0, Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(project_name)
        panel_layout.addWidget(project_subtitle)
        panel_layout.addSpacing(6)
        panel_layout.addWidget(developer_heading)
        panel_layout.addWidget(developer_text)
        panel_layout.addSpacing(4)
        panel_layout.addWidget(guide_heading)
        panel_layout.addWidget(guide_text)
        panel_layout.addSpacing(4)
        panel_layout.addWidget(hod_heading)
        panel_layout.addWidget(hod_text)
        panel_layout.addSpacing(10)
        panel_layout.addWidget(department_text)

        self.about_content_layout.addWidget(self.about_panel, 0, Qt.AlignmentFlag.AlignHCenter)
        scroll_layout.addLayout(self.about_content_layout)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        return page

    def create_card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        return card, layout

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f141b;
                color: #e8edf5;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            QStatusBar {
                background-color: #0b1016;
                color: #95a7bb;
            }
            QFrame#sidebar {
                background-color: #0b1016;
                border-right: 1px solid #1d2632;
            }
            QLabel#brandTitle {
                font-size: 22px;
                font-weight: 700;
                line-height: 1.2;
            }
            QLabel#brandSubtitle {
                color: #8ea2b6;
                font-size: 12px;
            }
            QPushButton#sidebarButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 14px;
                color: #c8d3df;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 14px;
                text-align: left;
            }
            QPushButton#sidebarButton:hover {
                background-color: #141b24;
                border-color: #243140;
            }
            QPushButton#sidebarButton:checked {
                background-color: #192534;
                border-color: #2f87c8;
                color: #ffffff;
            }
            QLabel#pageTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f3f7fb;
            }
            QLabel#pageSubtitle {
                color: #8fa3b7;
                font-size: 13px;
            }
            QFrame#card {
                background-color: #161d26;
                border: 1px solid #232d39;
                border-radius: 15px;
                padding: 10px;
            }
            QLabel#cardTitle {
                font-size: 18px;
                font-weight: 700;
                color: #f1f6fb;
            }
            QLabel#sectionLabel {
                font-size: 12px;
                font-weight: 700;
                color: #89a0b8;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            QLabel#resultTitle {
                font-size: 28px;
                font-weight: bold;
                color: #ffffff;
            }
            QLabel#bodyText {
                color: #d3dce6;
                line-height: 1.4;
            }
            QLabel#metricText {
                color: #f0f6fc;
                font-size: 18px;
                font-weight: 600;
                padding: 2px 0;
            }
            QLabel#mutedLabel {
                color: #9fb0c1;
                line-height: 1.45;
            }
            QLabel#imagePreview {
                background-color: #121922;
                border: 1px dashed #354559;
                border-radius: 24px;
                color: #7e90a4;
                font-size: 16px;
                padding: 18px;
            }
            QLabel#logoPlaceholder {
                background-color: #121922;
                border: 1px dashed #354559;
                border-radius: 18px;
                color: #7e90a4;
                font-size: 15px;
            }
            QFrame#aboutPanel {
                background-color: transparent;
                border: none;
            }
            QLabel#aboutCollegeName {
                color: #05d7ff;
                font-size: 25px;
                font-weight: 800;
                line-height: 1.25;
            }
            QLabel#aboutLogo {
                background-color: transparent;
                border: none;
                color: #7e90a4;
                font-size: 15px;
            }
            QLabel#aboutProjectName {
                color: #ffffff;
                font-size: 24px;
                font-weight: 800;
                line-height: 1.2;
            }
            QLabel#aboutProjectSubtitle {
                color: #f4f8fc;
                font-size: 18px;
                font-weight: 700;
                line-height: 1.3;
            }
            QLabel#aboutSectionHeading {
                color: #f6fbff;
                font-size: 15px;
                font-weight: 700;
                padding-top: 2px;
            }
            QLabel#aboutInfoText {
                color: #d9e4ef;
                font-size: 14px;
                line-height: 1.5;
            }
            QLabel#aboutFooterText {
                color: #dbe6f0;
                font-size: 15px;
                font-weight: 600;
                line-height: 1.4;
            }
            QLabel#statusBadge {
                background-color: #1a2330;
                border: 1px solid #2f3f53;
                border-radius: 10px;
                color: #dbe5ef;
                font-size: 14px;
                font-weight: 700;
                padding: 10px;
            }
            QPushButton#primaryButton,
            QPushButton#secondaryButton {
                min-height: 44px;
                border-radius: 14px;
                font-size: 14px;
                font-weight: 700;
                padding: 0 18px;
            }
            QPushButton#primaryButton {
                background-color: #6ddc7b;
                color: #0c1810;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background-color: #80e68c;
            }
            QPushButton#primaryButton:disabled {
                background-color: #2b3a2f;
                color: #7f9384;
            }
            QPushButton#secondaryButton {
                background-color: #1a2330;
                color: #edf4fa;
                border: 1px solid #2b3a4c;
            }
            QPushButton#secondaryButton:hover {
                background-color: #1d2a39;
            }
            QPushButton#secondaryButton:disabled {
                color: #7f8e9e;
                border-color: #25303c;
            }
            QProgressBar {
                background-color: #10161e;
                border: 1px solid #273240;
                border-radius: 11px;
                color: #ffffff;
                min-height: 24px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #45c972;
                border-radius: 10px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            """
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_responsive_layouts()

    def update_responsive_layouts(self) -> None:
        compact = self.width() < 1180

        self.home_main_split.setDirection(
            QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
        )

    def load_runtime_information(self) -> None:
        self.class_names = load_class_names(CLASS_INDICES_PATH)

        if self.class_names:
            self.class_count_label.setText(f"Number of Classes: {len(self.class_names)}")
        else:
            train_dir = DATASET_DIR / "train"
            class_count = len([item for item in train_dir.iterdir() if item.is_dir()]) if train_dir.exists() else 0
            self.class_count_label.setText(f"Number of Classes: {class_count if class_count else '--'}")

        split_counts = {
            "train": count_images(DATASET_DIR / "train"),
            "val": count_images(DATASET_DIR / "val"),
            "test": count_images(DATASET_DIR / "test"),
        }
        total_images = sum(split_counts.values())

        if total_images:
            self.image_count_label.setText(f"Number of Images: {total_images:,}")
            self.dataset_split_label.setText(
                f"Split Details: Train {split_counts['train']:,} | Val {split_counts['val']:,} | Test {split_counts['test']:,}"
            )
        else:
            self.image_count_label.setText("Number of Images: --")
            self.dataset_split_label.setText("Split Details: --")

        self.load_saved_metrics()
        self.load_default_logo()
        self.update_responsive_layouts()

    def load_saved_metrics(self) -> None:
        if not METRICS_PATH.exists():
            return

        try:
            with METRICS_PATH.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return

        self.set_model_metrics(
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
        )

    def set_model_metrics(
        self,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
    ) -> None:
        self.accuracy_label.setText(f"Accuracy: {format_metric(accuracy)}")
        self.precision_label.setText(f"Precision: {format_metric(precision)}")
        self.recall_label.setText(f"Recall: {format_metric(recall)}")
        self.f1_score_label.setText(f"F1 Score: {format_metric(f1_score)}")

    def run_model_evaluation(self) -> None:
        if self.evaluation_process is not None:
            return

        if not EVALUATION_SCRIPT_PATH.exists():
            QMessageBox.critical(self, "Evaluation Error", f"Evaluation script not found:\n{EVALUATION_SCRIPT_PATH}")
            return

        self.evaluation_process = QProcess(self)
        self.evaluation_process.setProgram(resolve_python_executable())
        self.evaluation_process.setArguments([str(EVALUATION_SCRIPT_PATH)])
        self.evaluation_process.setWorkingDirectory(str(BASE_DIR))
        self.evaluation_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.evaluation_process.finished.connect(self.on_model_evaluation_finished)

        self.evaluate_model_button.setEnabled(False)
        self.evaluate_model_button.setText("Evaluating...")
        self.performance_note_label.setText("Model evaluation is running. Scores will update automatically after completion.")
        self.statusBar().showMessage("Running model evaluation...", 0)
        self.evaluation_process.start()

    def on_model_evaluation_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        process = self.evaluation_process
        output_text = ""
        if process is not None:
            output_text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()

        self.evaluation_process = None
        self.evaluate_model_button.setEnabled(True)
        self.evaluate_model_button.setText("Evaluate Model")

        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.load_saved_metrics()
            self.performance_note_label.setText("Model evaluation completed. Performance scores have been updated.")
            self.statusBar().showMessage("Model evaluation completed", 5000)
            return

        self.performance_note_label.setText("Model evaluation failed. Please check the environment and dataset paths.")
        self.statusBar().showMessage("Model evaluation failed", 5000)
        QMessageBox.critical(
            self,
            "Evaluation Error",
            output_text or "Model evaluation could not be completed.",
        )

    def load_default_logo(self) -> None:
        if COLLEGE_LOGO_PATH.exists():
            self.set_college_logo(COLLEGE_LOGO_PATH)
            return

        for file_name in (
            "college_logo.png",
            "college_logo.jpg",
            "college_logo.jpeg",
            "logo.png",
            "logo.jpg",
        ):
            logo_path = BASE_DIR / file_name
            if logo_path.exists():
                self.set_college_logo(logo_path)
                return

    def set_college_logo(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            250,
            250,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.college_logo_label.setPixmap(scaled)
        self.college_logo_label.setText("")

    def reset_result_card(self) -> None:
        self.result_disease_label.setText("No prediction yet")
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("--")
        self.solution_label.setText("Solution will appear here")
        self.set_status_badge("Status")

    def set_status_badge(self, text: str) -> None:
        normalized = text.lower()
        if normalized == "healthy":
            background = "#153724"
            border = "#2c8e54"
            color = "#8ef0af"
        elif normalized == "diseased":
            background = "#3a1f1f"
            border = "#c75a5a"
            color = "#ff9f9f"
        elif normalized == "running model":
            background = "#2b2413"
            border = "#b38a2a"
            color = "#f3d27d"
        else:
            background = "#1a2330"
            border = "#324254"
            color = "#dce7f2"

        self.status_value_label.setText(text)
        self.status_value_label.setStyleSheet(
            f"""
            QLabel {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 10px;
                color: {color};
                font-size: 14px;
                font-weight: 700;
                padding: 10px;
            }}
            """
        )

    def switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if 0 <= index < len(self.nav_buttons):
            self.nav_buttons[index].setChecked(True)
            self.statusBar().showMessage(f"{self.nav_buttons[index].text()} page", 3000)

    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Leaf Image",
            str(BASE_DIR),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )

        if not file_path:
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Invalid Image", "The selected file could not be opened as an image.")
            return

        self.selected_image_path = Path(file_path)
        self.image_preview.set_preview_pixmap(pixmap)
        self.predict_button.setEnabled(True)
        self.reset_result_card()
        self.statusBar().showMessage(f"Loaded image: {self.selected_image_path.name}", 5000)

    def predict_disease(self) -> None:
        if not self.selected_image_path:
            QMessageBox.information(self, "Select Image", "Please select a leaf image before prediction.")
            return

        self.select_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.predict_button.setText("Predicting...")
        self.result_disease_label.setText("Analyzing image...")
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("Processing...")
        self.solution_label.setText("Loading model and running prediction on the selected image.")
        self.set_status_badge("Running Model")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            self.load_model_if_needed()
            result = self.run_prediction(self.selected_image_path)
            self.show_prediction_result(result)
            self.statusBar().showMessage("Prediction complete", 5000)
        except Exception as exc:
            self.reset_result_card()
            QMessageBox.critical(self, "Prediction Error", str(exc))
            self.statusBar().showMessage("Prediction failed", 5000)
        finally:
            QApplication.restoreOverrideCursor()
            self.select_button.setEnabled(True)
            self.predict_button.setEnabled(True)
            self.predict_button.setText("Predict")

    def load_model_if_needed(self) -> None:
        if self.model is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow and NumPy are required for prediction.") from exc

        self.np = np
        self.tf = tf
        self.model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

        if not self.class_names:
            self.class_names = load_class_names(CLASS_INDICES_PATH)

        if not self.class_names:
            output_units = int(self.model.output_shape[-1])
            self.class_names = [f"Class {index}" for index in range(output_units)]

    def run_prediction(self, image_path: Path) -> dict[str, object]:
        image = self.tf.keras.utils.load_img(str(image_path), target_size=IMAGE_SIZE)
        image_array = self.tf.keras.utils.img_to_array(image).astype("float32") / 255.0
        image_array = self.np.expand_dims(image_array, axis=0)

        predictions = self.model.predict(image_array, verbose=0)[0]
        predicted_index = int(self.np.argmax(predictions))
        confidence = float(predictions[predicted_index] * 100)

        raw_label = self.class_names[predicted_index] if predicted_index < len(self.class_names) else f"Class {predicted_index}"
        normalized_label = normalize_key(raw_label)
        disease_record = load_disease_solutions().get(normalized_label)

        if disease_record:
            plant_name = str(disease_record.get("plant", "")).strip()
            disease_name = str(disease_record.get("disease", "")).strip()
            display_label = " ".join(part for part in (plant_name, disease_name) if part).strip()
            solution_text = get_solution(raw_label)
        else:
            display_label = format_prediction_label(raw_label)
            solution_text = suggested_solution(raw_label)

        return {
            "raw_label": raw_label,
            "display_label": display_label,
            "confidence": confidence,
            "status": "Healthy" if is_healthy_label(raw_label) else "Diseased",
            "solution": solution_text,
        }

    def show_prediction_result(self, result: dict[str, object]) -> None:
        confidence = float(result["confidence"])
        progress_value = max(0, min(100, int(round(confidence))))

        self.result_disease_label.setText(str(result["display_label"]))
        self.confidence_bar.setValue(progress_value)
        self.confidence_bar.setFormat(f"{confidence:.2f}%")
        self.set_status_badge(str(result["status"]))
        self.solution_label.setText(str(result["solution"]))


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
