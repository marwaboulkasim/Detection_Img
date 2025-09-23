import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from prepare_data.data_cleaner import (
    filter_images_without_annotations,
    annotations_without_image,
    detect_bbox_anomalies,
    clean_annotations
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

def test_filter_images_without_annotations_empty_images_df():
    empty_images = pd.DataFrame(columns=["id", "file_name"])
    result = filter_images_without_annotations(empty_images, df_annotations)
    assert result.empty

def test_filter_images_without_annotations_all_returned():
    empty_annotations = pd.DataFrame(columns=["id", "image_id", "bbox"])
    result = filter_images_without_annotations(df_images, empty_annotations)
    assert len(result) == len(df_images)

def test_filter_images_without_annotations_partial():
    result = filter_images_without_annotations(df_images, df_annotations)
    assert len(result) == 1
    assert result.iloc[0]["id"] == 3

def test_annotations_without_image_empty_annotations():
    empty_annotations = pd.DataFrame(columns=["id", "image_id", "bbox"])
    result = annotations_without_image(empty_annotations, df_images)
    assert result.empty

def test_annotations_without_image_detects_missing():
    result = annotations_without_image(df_annotations, df_images)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == 99

def test_detect_bbox_anomalies_empty():
    empty_annotations = pd.DataFrame(columns=["bbox"])
    result = detect_bbox_anomalies(empty_annotations)
    assert result.empty

def test_detect_bbox_anomalies_detects():
    result = detect_bbox_anomalies(df_annotations)
    assert len(result) == 1
    assert result.iloc[0]["bbox"] == [5, 5, 0, 20]

def test_clean_annotations_removes_bad_and_missing():
    clean = clean_annotations(df_annotations, df_images)
    # bbox anormale supprimée, annotation sans image supprimée
    assert len(clean) == 1
    assert clean.iloc[0]["bbox"] == [10, 10, 50, 50]
