from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_TXT_PATH = BASE_DIR / "plants_disease_solution.txt"
OUTPUT_JSON_PATH = BASE_DIR / "disease_solutions.json"


PLANT_DISEASES = {
    "Apple": [
        "apple_scab",
        "apple_black_rot",
        "apple_cedar_apple_rust",
        "apple_healthy",
    ],
    "Blueberry": [
        "blueberry_healthy",
    ],
    "Cherry": [
        "cherry_powdery_mildew",
        "cherry_healthy",
    ],
    "Corn": [
        "corn_cercospora_leaf_spot",
        "corn_common_rust",
        "corn_northern_leaf_blight",
        "corn_healthy",
    ],
    "Grape": [
        "grape_black_rot",
        "grape_esca",
        "grape_leaf_blight",
        "grape_healthy",
    ],
    "Orange": [
        "orange_huanglongbing",
    ],
    "Peach": [
        "peach_bacterial_spot",
        "peach_healthy",
    ],
    "Pepper Bell": [
        "pepper_bell_bacterial_spot",
        "pepper_bell_healthy",
    ],
    "Potato": [
        "potato_early_blight",
        "potato_late_blight",
        "potato_healthy",
    ],
    "Strawberry": [
        "strawberry_leaf_scorch",
        "strawberry_healthy",
    ],
    "Squash": [
        "squash_powdery_mildew",
    ],
    "Soybean": [
        "soybean_healthy",
    ],
    "Raspberry": [
        "raspberry_healthy",
    ],
    "Tomato": [
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_septoria_leaf_spot",
        "tomato_spider_mites",
        "tomato_target_spot",
        "tomato_mosaic_virus",
        "tomato_yellow_leaf_curl_virus",
        "tomato_healthy",
    ],
}


LABEL_ALIASES = {
    "cherry_including_sour_powdery_mildew": "cherry_powdery_mildew",
    "cherry_including_sour_healthy": "cherry_healthy",
    "corn_maize_cercospora_leaf_spot_gray_leaf_spot": "corn_cercospora_leaf_spot",
    "corn_maize_common_rust": "corn_common_rust",
    "corn_maize_northern_leaf_blight": "corn_northern_leaf_blight",
    "corn_maize_healthy": "corn_healthy",
    "grape_esca_black_measles": "grape_esca",
    "grape_leaf_blight_isariopsis_leaf_spot": "grape_leaf_blight",
    "orange_haunglongbing_citrus_greening": "orange_huanglongbing",
    "orange_huanglongbing_citrus_greening": "orange_huanglongbing",
    "pepper_bell_bacterial_spot": "pepper_bell_bacterial_spot",
    "pepper_bell_healthy": "pepper_bell_healthy",
    "tomato_tomato_mosaic_virus": "tomato_mosaic_virus",
    "tomato_tomato_yellow_leaf_curl_virus": "tomato_yellow_leaf_curl_virus",
    "tomato_spider_mites_two_spotted_spider_mite": "tomato_spider_mites",
}


def normalize_key(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("(", " ")
    normalized = normalized.replace(")", " ")
    normalized = re.sub(r"[^a-z0-9_ ]+", "", normalized)
    normalized = re.sub(r"[\s_]+", "_", normalized).strip("_")
    normalized = normalized.replace("haunglongbing", "huanglongbing")

    tokens = [token for token in normalized.split("_") if token]
    deduped: list[str] = []
    for token in tokens:
        if not deduped or deduped[-1] != token:
            deduped.append(token)

    normalized = "_".join(deduped)
    return LABEL_ALIASES.get(normalized, normalized)


def readable_disease_name(plant_name: str, label_key: str) -> str:
    plant_key = normalize_key(plant_name)
    disease_key = label_key
    prefix = f"{plant_key}_"
    if disease_key.startswith(prefix):
        disease_key = disease_key[len(prefix) :]

    return disease_key.replace("_", " ").title()


def load_solution_reference(file_path: Path = SOURCE_TXT_PATH) -> dict[str, str]:
    solutions: dict[str, str] = {}
    if not file_path.exists():
        return solutions

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 3:
            continue

        plant_name, disease_name, solution_text = columns[:3]

        if disease_name.lower() == "disease" or set(disease_name) == {"-"}:
            continue
        if solution_text.lower() == "solution" or set(solution_text) == {"-"}:
            continue

        label_key = normalize_key(f"{plant_name}_{disease_name}")
        solutions[label_key] = solution_text

    return solutions


def rule_based_solution(label_key: str) -> str:
    disease_key = label_key

    if disease_key.endswith("_healthy"):
        return "Maintain regular care, balanced nutrition, proper watering, and routine monitoring."
    if "virus" in disease_key or "huanglongbing" in disease_key:
        return "Remove infected plants, control insect vectors, and monitor nearby plants to prevent spread."
    if "bacterial" in disease_key:
        return "Apply copper spray, remove infected leaves, and maintain field hygiene."
    if "spider_mites" in disease_key or "mites" in disease_key:
        return "Use neem oil or an appropriate miticide and inspect the undersides of leaves regularly."
    if any(term in disease_key for term in ("blight", "mildew", "rust", "mold", "rot", "scab", "spot", "esca", "scorch")):
        return "Apply fungicide, remove infected leaves, and improve airflow around the plant."

    return "Monitor the plant, remove affected leaves, and follow crop-specific disease management practices."


def build_disease_solutions() -> dict[str, dict[str, str]]:
    reference_solutions = load_solution_reference()
    dataset: dict[str, dict[str, str]] = {}

    for plant_name, disease_keys in PLANT_DISEASES.items():
        for disease_key in disease_keys:
            label_key = normalize_key(disease_key)
            solution_text = reference_solutions.get(label_key, rule_based_solution(label_key))

            dataset[label_key] = {
                "plant": plant_name,
                "disease": readable_disease_name(plant_name, label_key),
                "solution": solution_text,
            }

    return dataset


def save_disease_solutions(output_path: Path = OUTPUT_JSON_PATH) -> Path:
    dataset = build_disease_solutions()
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(dataset)} entries in {output_path.name}")
    return output_path


@lru_cache(maxsize=1)
def load_disease_solutions(json_path: Path = OUTPUT_JSON_PATH) -> dict[str, dict[str, str]]:
    if not json_path.exists():
        save_disease_solutions(json_path)

    return json.loads(json_path.read_text(encoding="utf-8"))


def get_solution(label: str) -> str:
    normalized_label = normalize_key(label)
    dataset = load_disease_solutions()

    record = dataset.get(normalized_label)
    if record:
        return record["solution"]

    return "Solution not found for this disease label."


# PySide6 GUI integration example:
# from build_disease_solutions import get_solution
#
# result_label = model_prediction_label
# solution_text = get_solution(result_label)
# self.solution_label.setText(solution_text)


if __name__ == "__main__":
    save_disease_solutions()
