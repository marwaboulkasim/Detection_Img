import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import json
from prepare_data.data_cleaner import (
    annotations_without_image,
    detect_bbox_anomalies,
    clean_annotations,
    get_images_without_annotations
)

# === Faux DataFrames pour tests sur annotations/images existantes ===
df_images = pd.DataFrame([
    {"id": 1, "file_name": "img1.jpg"},
    {"id": 2, "file_name": "img2.jpg"},
    {"id": 3, "file_name": "img3.jpg"},
])

df_annotations = pd.DataFrame([
    {"id": 10, "image_id": 1, "bbox": [10, 10, 50, 50]},
    {"id": 11, "image_id": 2, "bbox": [5, 5, 0, 20]},  # bbox anormale
    {"id": 12, "image_id": 99, "bbox": [0, 0, 10, 10]}, # annotation sans image
])

# === Tests pour les fonctions basées sur DataFrames ===
def test_annotations_without_image_detects_missing():
    result = annotations_without_image(df_annotations, df_images)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == 99

def test_detect_bbox_anomalies_detects():
    result = detect_bbox_anomalies(df_annotations)
    assert len(result) == 1
    assert result.iloc[0]["bbox"] == [5, 5, 0, 20]

def test_clean_annotations_removes_bad_and_missing():
    clean = clean_annotations(df_annotations, df_images)
    assert len(clean) == 1
    assert clean.iloc[0]["bbox"] == [10, 10, 50, 50]

# === Tests pour get_images_without_annotations avec fichiers temporaires ===
@pytest.fixture
def tmp_images_folder(tmp_path):
    # Créer un faux dossier avec images
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        (tmp_path / name).touch()
    return tmp_path

@pytest.fixture
def tmp_json_file(tmp_path):
    # Créer un faux fichier JSON COCO
    json_file = tmp_path / "_annotations.json"
    coco_dict = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"},
            {"id": 3, "file_name": "img3.jpg"}
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "bbox": [10,10,50,50]},
            {"id": 11, "image_id": 2, "bbox": [5,5,0,20]}  # bbox anormale
        ]
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)
    return json_file

def test_get_images_without_annotations_partial(tmp_images_folder, tmp_json_file):

    result = get_images_without_annotations(tmp_images_folder, tmp_json_file)
    # img3 n'a pas d'annotation => doit être retournée
    assert "img3.jpg" in result
    assert "img1.jpg" not in result

def test_get_images_without_annotations_all_returned(tmp_images_folder):
    # JSON vide => toutes les images doivent être retournées
    json_file = tmp_images_folder / "empty.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"images": [], "annotations": []}, f)
    result = get_images_without_annotations(tmp_images_folder, json_file)
    # Toutes les images du dossier sont retournées
    assert set(result) == {"img1.jpg", "img2.jpg", "img3.jpg"}

def test_get_images_without_annotations_empty_folder(tmp_path):
    # Dossier vide + JSON vide
    json_file = tmp_path / "empty.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"images": [], "annotations": []}, f)
    result = get_images_without_annotations(tmp_path, json_file)
    assert len(result) == 0
