import os
import shutil
import json
import random
import glob

# --- CONFIG ---
DATASET_DIR = "data"      # dossier contenant toutes les images et annotations
OUTPUT_DIR = "dataset"    # dossier de sortie
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
ANNOTATION_FILE = "_annotations.coco.json"
RANDOM_SEED = 42

# --- FONCTIONS --- Conversion des annotations COCO → YOLO
def convert_coco_to_yolo(annotation, img_width, img_height):
    x, y, w, h = annotation['bbox']
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    class_id = annotation['category_id']
    return [class_id, x_center, y_center, w_norm, h_norm]


#Organisation des dossiers
def prepare_dirs(output_dir):
    for split in SPLITS.keys():
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

#Découpage du dataset en train/val/test
def split_dataset(file_list):
    random.seed(RANDOM_SEED)
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(SPLITS['train'] * n)
    val_end = train_end + int(SPLITS['val'] * n)
    return {
        "train": file_list[:train_end],
        "val": file_list[train_end:val_end],
        "test": file_list[val_end:]
    }

# --- MAIN ---
if __name__ == "__main__":
    prepare_dirs(OUTPUT_DIR)

    # Charger annotations COCO
    with open(os.path.join(DATASET_DIR, ANNOTATION_FILE), 'r') as f:
        coco = json.load(f)

    images_info = {img['id']: img for img in coco['images']}
    annotations_per_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        annotations_per_image.setdefault(img_id, []).append(ann)

    # --- FILTRER LES IMAGES RÉELLEMENT PRÉSENTES ---
    found_images = []
    for img in images_info.values():
        base_name = img['file_name'].split('_rf.')[0]  # ignorer le hash
        # Chercher n'importe quel fichier contenant base_name
        matches = glob.glob(os.path.join(DATASET_DIR, f"*{base_name}*"))
        if matches:
            found_images.append(img)

    image_files = found_images
    dataset_split = split_dataset(image_files)

    # --- COPIE DES IMAGES ET CREATION DES LABELS ---
    for split, images in dataset_split.items():
        print(f"{split}: {len(images)} images")
        for img_info in images:
            img_name = img_info['file_name']
            img_id = img_info['id']

            base_name = img_name.split('_rf.')[0]
            matches = glob.glob(os.path.join(DATASET_DIR, f"*{base_name}*"))
            if not matches:
                print(f" Fichier introuvable pour {img_name}")
                continue

            img_path = matches[0]  # prend le premier fichier correspondant

            # Copier l'image
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, split, "images", os.path.basename(img_path)))

            # Créer le label YOLO
            label_path = os.path.join(OUTPUT_DIR, split, "labels", os.path.basename(img_path).replace('.jpg', '.txt'))
            anns = annotations_per_image.get(img_id, [])
            with open(label_path, 'w') as f:
                for ann in anns:
                    yolo_ann = convert_coco_to_yolo(ann, img_info['width'], img_info['height'])
                    f.write(" ".join([str(round(a,6)) for a in yolo_ann]) + "\n")

    print("Préparation du dataset terminée !")
