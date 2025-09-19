from pathlib import Path
import pandas as pd
import json
from collections import Counter

# Fonction 1 : récupérer les extensions des fichiers
def get_file_extensions(folder_path: str):
    folder = Path(folder_path)
    extensions = [file.suffix.lower() for file in folder.iterdir() if file.is_file()]
    return Counter(extensions)
img = get_file_extensions("./data")



# Fonction 2 : charger un fichier 
def load_coco_json(json_path: str):
    if not Path(json_path).is_file():
        raise FileNotFoundError(f"Le fichier JSON n'existe pas : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_dict = json.load(f)

    images_df = pd.DataFrame(coco_dict["images"])
    annotations_df = pd.DataFrame(coco_dict["annotations"])
    return images_df, annotations_df





# Fonction 3 : vérifier cohérence images vs JSON
def check_image(images_folder: str, images_df: pd.DataFrame):
    valid_exts = {".jpg", ".jpeg", ".png"}
    folder_files = {p.name for p in Path(images_folder).glob("*.*") if p.suffix.lower() in valid_exts}
    json_files = set(images_df["file_name"].tolist())

    missing_in_folder = json_files - folder_files
    missing_in_json = folder_files - json_files
    return missing_in_folder, missing_in_json



if __name__ == "__main__":
    path = "./data"
    print("Extensions trouvées :", get_file_extensions(path))

    
    json_path = "./data/_annotations.coco.json"
    images_df, annotations_df = load_coco_json(json_path)

   
    missing_in_folder, missing_in_json = check_image(path, images_df)
    print("Fichiers listés dans le JSON mais absents du dossier :", missing_in_folder)
    print("Fichiers présents dans le dossier mais absents du JSON :", missing_in_json)
