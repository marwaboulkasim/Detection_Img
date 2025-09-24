import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from prepare_data.data_cleaner import (
    annotations_without_image,
    detect_bbox_anomalies,
    clean_annotations,
    get_images_without_annotations
)

# === Faux DataFrames pour tests ===
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

# === Tests ===

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
    # bbox anormale et annotation sans image supprimées
    assert len(clean) == 1
    assert clean.iloc[0]["bbox"] == [10, 10, 50, 50]

def test_get_images_without_annotations_empty_images_df():
    empty_images = pd.DataFrame(columns=["id", "file_name"])
    result = get_images_without_annotations(empty_images, df_annotations)
    # Aucun fichier dans empty_images => résultat vide
    assert result.empty

def test_get_images_without_annotations_all_returned():
    empty_annotations = pd.DataFrame(columns=["id", "image_id", "bbox"])
    result = get_images_without_annotations(df_images, empty_annotations)
    # Toutes les images n'ont pas d'annotations
    assert len(result) == len(df_images)
    assert set(result) == set(df_images["file_name"])

def test_get_images_without_annotations_partial(tmp_path):
    # Créer un faux dossier avec images
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        (tmp_path / name).touch()

    # Créer un fichier JSON simulant les annotations COCO
    json_file = tmp_path / "annotations.json"
    coco_dict = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"},
            {"id": 3, "file_name": "img3.jpg"}
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "bbox": [10,10,50,50]},
            {"id": 11, "image_id": 2, "bbox": [5,5,0,20]}
        ]
    }
    import json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f)

    result = get_images_without_annotations(tmp_path, json_file)
    # img3 n'a pas d'annotation => doit être retournée
    assert "img3.jpg" in result
    assert "img1.jpg" not in result

