import json
import pandas as pd
from pathlib import Path


# Base directory (root of the project)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JSON_FILE = DATA_DIR / "_annotations.coco.json"


def load_coco_json(json_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge un fichier COCO JSON et renvoie les DataFrames images, annotations et categories.
    """
    json_file = Path(json_path)
    if not json_file.is_file():
        raise FileNotFoundError(f"Fichier JSON introuvable : {json_file}")
    
    with json_file.open('r', encoding='utf-8') as f:
        coco_dict = json.load(f)
    # print(coco_dict)
    images_df = pd.DataFrame(coco_dict.get("images", []))
    annotations_df = pd.DataFrame(coco_dict.get("annotations", []))
    categories_df = pd.DataFrame(coco_dict.get("categories", []))
    
    # Affichage résumé
    print(f"Nombre d'images : {len(images_df)}")
    print(f"Nombre d'annotations : {len(annotations_df)}")
    if not categories_df.empty and 'name' in categories_df.columns:
        print(f"Catégories : {categories_df['name'].tolist()}")
    else:
        print(f"Catégories : {categories_df.to_dict(orient='records')}")
    
    return images_df, annotations_df, categories_df

# Exemple d'utilisation
 #images_df, annotations_df, categories_df = load_coco_json("data/_annotations.coco.json")
