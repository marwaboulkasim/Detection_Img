from prepare_data.data_loader import load_coco_json
from prepare_data.data_explorer import annotations_per_image, images_without_annotations, images_per_category, bbox_stats
from prepare_data.data_cleaner import get_file_extensions, get_images_without_annotations, detect_bbox_anomalies, remove_images_without_annotations



def loader():
    images_df, annotations_df, categories_df = load_coco_json("data/_annotations.coco.json")

    return images_df, annotations_df, categories_df

def explorer():
    images_df, annotations_df, categories_df = loader()
    ann_per_img = annotations_per_image(annotations_df)
    #print(ann_per_img)

    img_no_ann = images_without_annotations(images_df, annotations_df)
    #print(img_no_ann)


    img_per_category = images_per_category(annotations_df, categories_df)
    #print(img_per_category)


    box_stats = bbox_stats(annotations_df)
    #print(box_stats)
    return ann_per_img, img_no_ann, img_per_category, box_stats

def cleaner():
    clean = get_file_extensions("data")
   # print(clean)

    
    filter_imgs_without_ann = get_images_without_annotations("data", "data/_annotations.coco.json")
    #print(f" list of images without annotation: {filter_imgs_without_ann}")

    images_df, annotations_df, categories_df = loader()
    detect_box_annom = detect_bbox_anomalies(annotations_df)
    #print(f" Détecte les BBoxes aberrantes{detect_box_annom}")


    imgs_no-anns = remove_images_without_annotations(images_df, annotations_df, "data")
    return clean, filter_imgs_without_ann, detect_box_annom




