import json
import pandas as pd

def json_loader():
    """
    Load a COCO-style JSON file and convert it into a Python dictionary.

    Returns:
        dict: The content of the JSON file as a Python dictionary.
    """
    # Open the JSON file in read mode
    with open('dataset/_annotations.coco.json', 'r') as f:
        # Parse the JSON file and convert it to a Python dictionary
        data = json.load(f)
        return data

# Call the function to load JSON data
new_data = json_loader()

# Print the top-level keys of the JSON dictionary
print(new_data.keys())
# Expected keys: ['info', 'licenses', 'categories', 'images', 'annotations']


def json_df(json_data):
    """
    Convert COCO-style JSON data into three Pandas DataFrames: categories, images, and annotations.

    Args:
        json_data (dict): JSON data loaded from a COCO-style annotation file.

    Returns:
        tuple: A tuple containing three DataFrames in the order:
            - df_categories: DataFrame of categories
            - df_images: DataFrame of images
            - df_annotations: DataFrame of annotations
    """
    # Convert the 'categories' section into a DataFrame
    df_categories = pd.DataFrame(json_data['categories'])

    # Convert the 'images' section into a DataFrame
    df_images = pd.DataFrame(json_data['images'])

    # Convert the 'annotations' section into a DataFrame
    df_annotations = pd.DataFrame(json_data['annotations'])

    return df_categories, df_images, df_annotations

# Call the function to convert JSON data into DataFrames
df_categories, df_images, df_annotations = json_df(new_data)

# Preview the DataFrames
print("Categories DataFrame:")
print(df_categories)

print("\nImages DataFrame:")
print(df_images)

print("\nAnnotations DataFrame:")
print(df_annotations)

    