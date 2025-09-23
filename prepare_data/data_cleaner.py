# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import json
import os
from PIL import Image
from collections import Counter

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

# === Fonctions de nettoyage et traitement des anomalies ===

def filter_images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les images qui n'ont aucune annotation"""
    unique_image_ids = annotations_df['image_id'].unique()
    images_no_ann = images_df[~images_df['id'].isin(unique_image_ids)]
    return images_no_ann

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
    images_no_ann = filter_images_without_annotations(images_df, annotations_df)
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
