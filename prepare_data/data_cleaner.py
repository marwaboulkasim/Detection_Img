# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import json
import os
from PIL import Image
from collections import Counter
from prepare_data.data_loader import load_coco_json

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
    """
    Return a list of images that have no annotations.
    """
    # Load COCO file
    with open(json_file, 'r', encoding='utf-8') as f:
        coco_dict = json.load(f)

    # Get all image IDs that have at least one annotation
    annotated_image_ids = {ann["image_id"] for ann in coco_dict["annotations"]}

    # Map image IDs to file names
    id_to_name = {img["id"]: img["file_name"] for img in coco_dict["images"]}
    annotated_images = {id_to_name[iid] for iid in annotated_image_ids}

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Get all images in the folder
    folder = Path(images_folder)
    all_images = {f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts}

    # Find images without annotations
    images_without_ann = set(all_images) - set(annotated_images)

    return images_without_ann

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

def remove_images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame, images_folder: Path) -> pd.DataFrame:
    """Supprime les images sans annotations physiquement et dans le DataFrame"""
    images_no_ann = get_images_without_annotations(images_df, annotations_df)
    for fname in images_no_ann['file_name']:
        remove_file(images_folder / fname)
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
