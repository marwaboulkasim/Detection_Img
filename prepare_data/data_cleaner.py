# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import json
import os
from PIL import Image
from collections import Counter
#from prepare_data.data_loader import load_coco_json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JSON_FILE = DATA_DIR / "_annotations.coco.json"

# === Fonctions utilitaires ===

def get_file_extensions(folder_path: str) -> Counter:
    """Retourne un compteur des extensions de fichiers dans un dossier"""
    folder = Path(folder_path)
    extensions = [file.suffix.lower() for file in folder.iterdir() if file.is_file()]
    return Counter(extensions)

def show_image(image_path: str, save_folder: str = None):
    """Affiche une image et éventuellement la sauvegarde dans un dossier de vérification"""
    if not Path(image_path).is_file():
        print(f"Fichier introuvable : {image_path}")
        return
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        dst_path = os.path.join(save_folder, os.path.basename(image_path))
        Image.open(image_path).save(dst_path)

def missing_values(df: pd.DataFrame) -> pd.Series:
    """Compte les valeurs manquantes d'un DataFrame"""
    return df.isna().sum()

def remove_file(file_path: Path):
    """Supprime un fichier s'il existe"""
    if file_path.is_file():
        os.remove(file_path)
        print(f"file has been removed successfuly ")
    else:
        print("Error, check the function!")

# === Fonctions de nettoyage et traitement des anomalies ===

def get_images_without_annotations(images_folder, json_file):
    # Ensure we are working with Path objects
    images_folder = Path(images_folder)
    json_file = Path(json_file)

    # Load COCO JSON
    with open(json_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Get annotated image filenames
    annotated = {
        img["file_name"]
        for img in coco.get("images", [])
        if any(ann["image_id"] == img["id"] for ann in coco.get("annotations", []))
    }

    # Get all images in the folder
    all_images = {p.name for p in images_folder.glob("*.jpg")}

    # Return images without annotations
    return list(all_images - annotated)

#Example usage
#result = get_images_without_annotations("../dataset", "../dataset/data/_annotations.coco.json")



def annotations_without_image(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les annotations dont les images n'existent pas"""
    valid_image_ids = images_df['id'].unique()
    return annotations_df[~annotations_df['image_id'].isin(valid_image_ids)]

def detect_bbox_anomalies(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les BBoxes aberrantes"""
    def is_abnormal(bbox):
        w, h = bbox[2], bbox[3]
        return w == 0 or h == 0
    anomalies = annotations_df[annotations_df['bbox'].apply(is_abnormal)]
    return anomalies

def remove_images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame, DATA_DIR) -> pd.DataFrame:
    """Supprime les images sans annotations physiquement et dans le DataFrame"""
    images_no_ann = get_images_without_annotations(images_df, annotations_df)
    for fname in images_no_ann['file_name']:
        remove_file(DATA_DIR / fname)
    images_df_clean = images_df[images_df['id'].isin(annotations_df['image_id'].unique())]
    return images_df_clean

def clean_annotations(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les annotations sans image et les BBoxes aberrantes"""
    bad_bboxes = detect_bbox_anomalies(annotations_df)
    annotations_df = annotations_df.drop(bad_bboxes.index, errors='ignore')
    missing_ann = annotations_without_image(annotations_df, images_df)
    annotations_df = annotations_df[annotations_df['image_id'].isin(images_df['id'])]
    return annotations_df

def save_coco_json(images_df: pd.DataFrame, annotations_df: pd.DataFrame, categories: list, save_path: Path):
    """Sauvegarde un fichier COCO JSON"""
    coco_clean = {
        "images": images_df.to_dict(orient="records"),
        "annotations": annotations_df.to_dict(orient="records"),
        "categories": categories
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(coco_clean, f, ensure_ascii=False, indent=4)
