import fiftyone as fo
from pathlib import Path

# Chemin vers ton dossier data
base_dir = Path("/home/marwa/Detection_Img/data")

# Images et annotations
images_dir = base_dir
annotations_path = base_dir / "_annotations_clean.coco.json"

# Charger le dataset COCO en précisant data_path vide
dataset = fo.Dataset.from_dir(
    dataset_dir=str(images_dir),
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=str(annotations_path),
    data_path=""  # <-- IMPORTANT : sinon FiftyOne cherche data/data/
)

# Lancer la visualisation interactive
session = fo.launch_app(dataset)
