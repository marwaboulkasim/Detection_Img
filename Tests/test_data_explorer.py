import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from prepare_data.data_explorer import (
    annotations_per_image,
    images_without_annotations,
    images_per_category,
    bbox_stats,
    bbox_issues
)

# === Faux DataFrames pour tests ===
df_images = pd.DataFrame([
    {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
    {"id": 2, "file_name": "img2.jpg", "width": 50, "height": 50},
    {"id": 3, "file_name": "img3.jpg", "width": 200, "height": 200},
])

df_annotations = pd.DataFrame([
    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]},
    {"id": 11, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},
    {"id": 12, "image_id": 2, "category_id": 1, "bbox": [5, 5, 0, 20]},  # bbox anormale
])

df_categories = pd.DataFrame([
    {"id": 1, "name": "catA"},
    {"id": 2, "name": "catB"},
])

# === Tests ===

def test_annotations_per_image_counts_correctly():
    result = annotations_per_image(df_annotations)
    img1_count = result[result["image_id"] == 1]["num_annotations"].iloc[0]
    assert img1_count == 2

def test_annotations_per_image_empty():
    empty = pd.DataFrame(columns=["image_id"])
    result = annotations_per_image(empty)
    assert result.empty

def test_images_without_annotations_detects():
    result = images_without_annotations(df_images, df_annotations)
    assert len(result) == 1
    assert result.iloc[0]["id"] == 3

def test_images_per_category_counts():
    result = images_per_category(df_annotations, df_categories)
    assert set(result["category"]) == {"catA", "catB"}
    catA_row = result[result["category"] == "catA"]
    assert catA_row["num_images"].iloc[0] == 2

def test_bbox_stats_contains_expected_columns():
    stats = bbox_stats(df_annotations)
    for col in ["width", "height", "area"]:
        assert col in stats.columns or col in stats.index.names

def test_bbox_issues_detects_invalid():
    result = bbox_issues(df_annotations, df_images)
    assert len(result) == 1
    assert result.iloc[0]["bbox"] == [5, 5, 0, 20]
