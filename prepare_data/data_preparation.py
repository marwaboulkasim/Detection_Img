from pathlib import Path
import pandas as pd
import json

# --- 1. Fonction COCO → YOLO ---
def convert_coco_to_yolo(annotations_df, images_df):
    img_size = images_df.set_index('id')[['width', 'height']]
    yolo_annotations = []

    for idx, row in annotations_df.iterrows():
        image_id = row['image_id']
        cat_id = row['category_id']
        bbox = row['bbox']
        if image_id not in img_size.index:
            continue
        img_w, img_h = img_size.loc[image_id]['width'], img_size.loc[image_id]['height']
        x_min, y_min, w, h = bbox
        x_center = (x_min + w / 2) / img_w
        y_center = (y_min + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        yolo_annotations.append({
            "image_id": image_id,
            "class": cat_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": w_norm,
            "height": h_norm
        })
    return pd.DataFrame(yolo_annotations)

# --- 2. Charger le COCO ---
coco_file = Path("data/_annotations.coco.json")
with open(coco_file, "r", encoding="utf-8") as f:
    coco = json.load(f)

images_df = pd.DataFrame(coco["images"])
annotations_df = pd.DataFrame(coco["annotations"])

# --- 3. Conversion ---
yolo_df = convert_coco_to_yolo(annotations_df, images_df)
print(yolo_df.head())
