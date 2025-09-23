from pathlib import Path
from prepare_data.data_loader import load_coco_json
from prepare_data.data_explorer import (
    annotations_per_image,
    images_without_annotations,
    images_per_category,
    bbox_stats,
    bbox_issues,
    missing_values
)
from prepare_data.data_cleaner import (
    remove_images_without_annotations,
    clean_annotations,
    save_coco_json,
    get_file_extensions
)


def run_pipeline(coco_json_path: str | Path = "data/_annotations.coco.json"):
    data_folder = Path("data")

    # === 1. Chargement ===
    images_df, annotations_df, categories_df = load_coco_json(coco_json_path)

    # === 2. Exploration ===
    print("\n--- Exploration des données ---\n")
    print("Nombre total d'images :", len(images_df))
    print("Nombre total d'annotations :", len(annotations_df))
    print("Catégories :", categories_df['name'].tolist())

    print("\nNombre d'images par catégorie :")
    print(images_per_category(annotations_df, categories_df))

    print("\nStatistiques sur le nombre d'annotations par image :")
    print(annotations_per_image(annotations_df)['num_annotations'].describe())

    print("\nQuelques images sans annotations :")
    print(images_without_annotations(images_df, annotations_df).head())

    print("\n--- Statistiques Bounding Boxes ---")
    print(bbox_stats(annotations_df))

    print("\n--- BBoxes problématiques (hors limites ou surface nulle) ---")
    invalid_bbox = bbox_issues(annotations_df, images_df)
    print(invalid_bbox[['image_id', 'bbox']].head())
    print("Nombre de bboxes problématiques :", len(invalid_bbox))

    print("\n--- Valeurs manquantes ---")
    print("Images :", missing_values(images_df))
    print("Annotations :", missing_values(annotations_df))
    print("Catégories :", missing_values(categories_df))

    print("\n--- Extensions détectées dans data/ ---")
    print(get_file_extensions(data_folder))

    # === 3. Nettoyage ===
    print("\n--- Nettoyage des données ---\n")
    images_df_clean = remove_images_without_annotations(images_df, annotations_df, data_folder)
    annotations_df_clean = clean_annotations(annotations_df, images_df_clean)

    clean_json_path = data_folder / "_annotations_clean.coco.json"
    save_coco_json(images_df_clean, annotations_df_clean, categories_df.to_dict(orient="records"), clean_json_path)
    print(f"JSON nettoyé sauvegardé dans : {clean_json_path}")


if __name__ == "__main__":
    run_pipeline("data/_annotations.coco.json")
