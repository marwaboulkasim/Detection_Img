import json
import pandas as pd

# === Charger le fichier COCO ===
with open("data/_annotations.coco.json", "r") as f:
    coco = json.load(f)

# Transformer en DataFrames
images_df = pd.DataFrame(coco["images"])
annotations_df = pd.DataFrame(coco["annotations"])
categories_df = pd.DataFrame(coco["categories"])

# === Fonctions d'exploration ===
def annotations_per_image(annotations_df: pd.DataFrame) -> pd.DataFrame:
    return annotations_df.groupby('image_id').size().reset_index(name='num_annotations')

def images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    annotated_ids = annotations_df['image_id'].unique()
    return images_df[~images_df['id'].isin(annotated_ids)]

def images_per_category(annotations_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    merged = annotations_df.merge(categories_df, left_on='category_id', right_on='id')
    counts = merged.groupby('name')['image_id'].nunique().reset_index(name='num_images')
    counts.columns = ['category', 'num_images']
    return counts.sort_values(by='num_images', ascending=False)

def bbox_stats(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Stats sur les bounding boxes (largeur, hauteur, surface)."""
    df = annotations_df.copy()
    df['width'] = df['bbox'].apply(lambda x: x[2])
    df['height'] = df['bbox'].apply(lambda x: x[3])
    df['area'] = df['width'] * df['height']
    return df[['width', 'height', 'area']].describe()

def bbox_issues(annotations_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """Detecte bboxes hors limites ou surface nulle."""
    merged = annotations_df.merge(images_df, left_on='image_id', right_on='id', suffixes=('_ann', '_img'))
    
    # Extraire les coordonnées
    merged['x_min'] = merged['bbox'].apply(lambda x: x[0])
    merged['y_min'] = merged['bbox'].apply(lambda x: x[1])
    merged['w'] = merged['bbox'].apply(lambda x: x[2])
    merged['h'] = merged['bbox'].apply(lambda x: x[3])
    
    # Conditions invalides
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
    """Nombre de valeurs manquantes par colonne."""
    return df.isna().sum()


# === Exploration avancée ===
# if __name__ == "__main__":
#     print("\n--- Exploration des données ---\n")

#     print("Nombre total d'images :", len(images_df))
#     print("Nombre total d'annotations :", len(annotations_df))
#     print("Catégories :", categories_df['name'].tolist())

#     print("\nNombre d'images par catégorie :")
#     print(images_per_category(annotations_df, categories_df))

#    # print("\nStatistiques sur le nombre d'annotations par image :")
#     #print(annotations_per_image(annotations_df)['num_annotations'].describe())

#     print("\nQuelques images sans annotations :")
#     print(images_without_annotations(images_df, annotations_df).head())

#     print("\n--- Statistiques Bounding Boxes ---")
#     print(bbox_stats(annotations_df))

#     print("\n--- BBoxes problématiques (hors limites ou surface nulle) ---")
#     invalid_bbox = bbox_issues(annotations_df, images_df)
#     print(invalid_bbox[['image_id', 'bbox', 'width', 'height', 'area']].head())
#     print("Nombre de bboxes problématiques :", len(invalid_bbox))

#     print("\n--- Valeurs manquantes ---")
#     print("Images :", missing_values(images_df))
#     print("Annotations :", missing_values(annotations_df))
#     print("Catégories :", missing_values(categories_df))
