import pandas as pd
from pathlib import Path
import json



def total_number(df):
    """
    Calculate the total number of rows in a DataFrame.

    This can be used to count the total number of images, annotations, or categories,
    depending on which DataFrame is passed.

    Args:
        df (pd.DataFrame): The DataFrame to count rows from.

    Returns:
        int: Total number of rows in the DataFrame.
    """
    return len(df)

# example iusage 
# total_number(df_images)
# print(f"Total number of images: {total_number(df_images)}")


def categories_names(df):
    """
    Return a list of category names from a categories DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the 'name' column for categories.

    Returns:
        list: List of category names.
    """
    return df['name'].tolist()
    # usage
    # print("Category names:")
    # print(categories_names(df_categories), "\n")


def images_per_category(df_annotations, df_categories):
    """
    Calculate the number of unique images per category.

    Args:
        df_annotations (pd.DataFrame): DataFrame containing 'image_id' and 'category_id'.
        df_categories (pd.DataFrame): DataFrame containing 'id' and 'name' for categories.

    Returns:
        pd.Series: Number of unique images per category.
    """
    merged = df_annotations.merge(df_categories, left_on='category_id', right_on='id')
    unique_images = merged[['image_id', 'name']].drop_duplicates()
    return unique_images.groupby('name').size()
    # example usage
    # print("Images per category:")
    # print(images_per_category(df_annotations, df_categories), "\n")

def annotations_statistics(df_annotations):
    """
    Compute the number of annotations per image and provide summary statistics.

    Args:
        df_annotations (pd.DataFrame): DataFrame containing annotations with a column 'image_id'.

    Returns:
        tuple:
            - annotations_per_image (pd.Series): Number of annotations for each image.
            - stats (pd.Series): Summary statistics of annotations per image (count, mean, min, max, quartiles).
    """
    annotations_per_image = df_annotations.groupby('image_id').size()
    stats = annotations_per_image.describe()
    return annotations_per_image, stats
    # example usage
    # annotations_per_image, stats = annotations_statistics(df_annotations)
    # print("Number of annotations per image (preview):")
    # print(annotations_per_image.head(), "\n")


def verify_images(df: pd.DataFrame, column_name: str, folder_path: str):
    """
    Verify consistency between image names in a DataFrame and actual images in a folder.
    """
    p = Path(folder_path)

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Collect only image file names in the folder
    n_names = [n.name for n in p.iterdir() if n.is_file() and n.suffix.lower() in valid_exts]

    # Get image names from DataFrame column
    img_names = df[column_name].to_list()

    # Convert both to sets
    set_folder = set(n_names)
    set_df = set(img_names)

    # Compare
    missing_in_folder = set_df - set_folder
    missing_in_df = set_folder - set_df
    if len(missing_in_folder) > 0 :
        print(f"there is {missing_in_folder} missed images in the folder" )
    else:
        print(f"there is no missing immages in the folder")
    if len(missing_in_df) > 0 :
        print(f"there is {missing_in_df} missed images in the dataframe" )
    else:
        print(f"there is no missing immages in the dataframe ")
    return missing_in_folder, missing_in_df




def get_images_without_annotations(images_folder, annotations_file):
    """
    Return a list of images that have no annotations.
    """
    # Load COCO file
    with open(annotations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get all image IDs that have at least one annotation
    annotated_image_ids = {ann["image_id"] for ann in data["annotations"]}

    # Map image IDs to file names
    id_to_name = {img["id"]: img["file_name"] for img in data["images"]}
    annotated_images = {id_to_name[iid] for iid in annotated_image_ids}

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Get all images in the folder
    folder = Path(images_folder)
    all_images = {f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts}

    # Find images without annotations
    images_without_ann = all_images - annotated_images
    if images_without_ann:
        print("Images without annotations are:", list(images_without_ann))
    else:
        print("There are no images without annotations.")

    return list(images_without_ann)

# # Example usage
# result = get_images_without_annotations("../dataset", "../dataset/data/_annotations.coco.json")



def detect_bbox_outliers(df: pd.DataFrame):
    """
    Detect outliers in the 'bbox' column:
    - width or height = 0
    - width or height < 0
    Returns a DataFrame of the outlier rows.
    """
    widths = df['bbox'].apply(lambda x: x[2])
    heights = df['bbox'].apply(lambda x: x[3])

    condition = (widths <= 0) | (heights <= 0)

    outliers = df[condition]

    return outliers
# example usage
# outlier_annotations = detect_bbox_outliers(df_annotations)