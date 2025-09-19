import json
import pandas as pd
from pathlib import Path

# Chemin vers le fichier JSON
json_file = Path("data/_annotations.coco.json")

# Chargement du JSON
with json_file.open('r', encoding='utf-8') as f:
    coco_dict = json.load(f)

# Conversion en DataFrames
images_df = pd.DataFrame(coco_dict.get("images", []))
annotations_df = pd.DataFrame(coco_dict.get("annotations", []))
categories_df = pd.DataFrame(coco_dict.get("categories", []))

# Affichage résumé
print(f"Nombre d'images : {len(images_df)}")
print(f"Nombre d'annotations : {len(annotations_df)}")
if 'name' in categories_df.columns:
    print(f"Catégories : {categories_df['name'].tolist()}")
else:
    print(f"Catégories : {categories_df.to_dict(orient='records')}")
