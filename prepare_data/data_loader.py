import json
import pandas as pd

def json_loader(filepath):
    """Load a COCO-style JSON file and convert it into a Python dictionary."""
    print(filepath)
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


