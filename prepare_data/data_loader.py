import json
import pandas as pd

def json_loader(filepath='dataset/_annotations.coco.json'):
    """Load a COCO-style JSON file and convert it into a Python dictionary."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def json_df(json_data):
    """
    Convert COCO-style JSON data into three Pandas DataFrames:
    categories, images, and annotations.
    """
    df_categories = pd.DataFrame(json_data.get('categories', []))
    df_images = pd.DataFrame(json_data.get('images', []))
    df_annotations = pd.DataFrame(json_data.get('annotations', []))
    return df_categories, df_images, df_annotations


# --- Load data globally (so it can be imported from other files) ---
_data = json_loader()
df_categories, df_images, df_annotations = json_df(_data)


if __name__ == "__main__":
    print("Categories DataFrame:")
    print(df_categories.head(), "\n")

    print("Images DataFrame:")
    print(df_images.head(), "\n")

    print("Annotations DataFrame:")
    print(df_annotations.head(), "\n")
