from pathlib import Path
import pandas as pd
from collections import Counter
import json
import os
from PIL import Image

# === Charger le JSON COCO ===
json_path = "data/_annotations.coco.json"
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]            # liste des images
annotations = coco["annotations"]  # liste des annotations

# === Fonctions utilitaires ===
def get_file_extensions(folder_path: str) -> Counter:
    folder = Path(folder_path)
    extensions = [file.suffix.lower() for file in folder.iterdir() if file.is_file()]
    return Counter(extensions)

def load_coco_json(json_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not Path(json_path).is_file():
        raise FileNotFoundError(f"Le fichier JSON n'existe pas : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_dict = json.load(f)
    images_df = pd.DataFrame(coco_dict["images"])
    annotations_df = pd.DataFrame(coco_dict["annotations"])
    return images_df, annotations_df

def check_image(images_folder: str, images_df: pd.DataFrame) -> tuple[set, set]:
    valid_exts = {".jpg", ".jpeg", ".png"}
    folder_files = {p.name for p in Path(images_folder).glob("*.*") if p.suffix.lower() in valid_exts}
    json_files = set(images_df["file_name"].tolist())
    missing_in_folder = json_files - folder_files
    missing_in_json = folder_files - json_files
    return missing_in_folder, missing_in_json

def check_bboxes(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    merged = annotations_df.merge(images_df, left_on='image_id', right_on='id', suffixes=('_ann', '_img'))
    merged['x_min'] = merged['bbox'].apply(lambda x: x[0])
    merged['y_min'] = merged['bbox'].apply(lambda x: x[1])
    merged['w'] = merged['bbox'].apply(lambda x: x[2])
    merged['h'] = merged['bbox'].apply(lambda x: x[3])
    invalid = merged[
        (merged['x_min'] < 0) |
        (merged['y_min'] < 0) |
        (merged['x_min'] + merged['w'] > merged['width']) |
        (merged['y_min'] + merged['h'] > merged['height']) |
        (merged['w'] == 0) |
        (merged['h'] == 0)
    ]
    return invalid

def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()

def filter_images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne les images qui n'ont aucune annotation.
    """
    # 1️⃣ Liste des IDs uniques des images annotées
    unique_image_ids = annotations_df['image_id'].unique()
    print(f"Nombre d'IDs uniques dans les annotations : {len(unique_image_ids)}")

    # 2️⃣ Filtrage du DataFrame images
    images_no_ann = images_df[~images_df['id'].isin(unique_image_ids)]
    
    print(f"Nombre d'images sans annotations : {len(images_no_ann)}")
    return images_no_ann


def show_image(image_path: str, save_folder: str = None):
    if not Path(image_path).is_file():
        print(f"Fichier introuvable : {image_path}")
        return
    print(f"Image prête à être vérifiée : {image_path}")
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        dst_path = os.path.join(save_folder, os.path.basename(image_path))
        Image.open(image_path).save(dst_path)
        print(f"Image sauvegardée dans : {dst_path}")

def annotations_without_image(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    valid_image_ids = images_df['id'].unique()
    return annotations_df[~annotations_df['image_id'].isin(valid_image_ids)]

def detect_bbox_anomalies(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les annotations avec des bboxes aberrantes :
        - largeur == 0 et hauteur != 0
        - largeur != 0 et hauteur == 0
        - largeur == 0 et hauteur == 0
    Args:
        annotations_df (pd.DataFrame): DataFrame contenant une colonne 'bbox' [x, y, w, h]
    Returns:
        pd.DataFrame: DataFrame filtré avec uniquement les bboxes aberrantes
    """
    def is_abnormal(bbox):
        w, h = bbox[2], bbox[3]
        # Valeurs aberrantes
        return (w == 0 and h != 0) or (w != 0 and h == 0) or (w == 0 and h == 0)
    
    anomalies = annotations_df[annotations_df['bbox'].apply(is_abnormal)]
    return anomalies

# === Script principal ===
if __name__ == "__main__":
    path = "./data"
    print("Extensions trouvées :", get_file_extensions(path))

    images_df, annotations_df = load_coco_json(json_path)
    missing_in_folder, missing_in_json = check_image(path, images_df)
    print("Fichiers listés dans le JSON mais absents du dossier :", missing_in_folder)
    print("Fichiers présents dans le dossier mais absents du JSON :", missing_in_json)

    invalid_bboxes = check_bboxes(annotations_df, images_df)
    print("Nombre de BBoxes invalides :", len(invalid_bboxes))

    print("Valeurs manquantes images :", missing_values(images_df))
    print("Valeurs manquantes annotations :", missing_values(annotations_df))

    images_no_ann = filter_images_without_annotations(images_df, annotations_df)
    print("Nombre d'images sans annotations :", len(images_no_ann))

    # Sauvegarder une image pour vérification
    if len(images_no_ann) > 0:
        first_image_path = os.path.join(path, images_no_ann.iloc[0]['file_name'])
        show_image(first_image_path, save_folder="./check_images")

    missing_ann = annotations_without_image(annotations_df, images_df)
    print("Annotations sans image :", len(missing_ann))

    bad_bboxes = detect_bbox_anomalies(annotations_df)
    print("BBoxes aberrantes :", len(bad_bboxes))
