import pandas as pd
from prepare_data.data_loader import images_df, annotations_df, categories_df



def annotations_per_image(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'annotations par image."""
    return annotations_df.groupby('image_id').size().reset_index()

def images_without_annotations(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Images sans annotations."""
    annotated_ids = annotations_df['image_id'].unique()
    return images_df[~images_df['id'].isin(annotated_ids)]

def images_per_category(annotations_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'images par catégorie."""
    merged = annotations_df.merge(categories_df, left_on='category_id', right_on='id')
    counts = merged.groupby('name')['image_id'].nunique().reset_index()
    counts.columns = ['category', 'num_images']
    return counts


print("\n--- Exploration des données ---\n")

# 1️ Nombre total d'images
total_images = len(images_df)
print("Nombre total d'images :", total_images)

# 2️ Nombre total d'annotations
total_annotations = len(annotations_df)
print("Nombre total d'annotations :", total_annotations)

# 3️ Catégories présentes
if 'name' in categories_df.columns:
    categories = categories_df['name'].tolist()
else:
    categories = []
print("Catégories :", categories)

# 4️ Nombre d'images par catégorie
print("\nNombre d'images par catégorie :")
print(images_per_category(annotations_df, categories_df))

# 5️ Stat sur le nombre d'annotations par image
annotations_count = annotations_df.groupby('image_id').size()

print("\nStatistiques sur le nombre d'annotations par image :")
print(annotations_count.describe())

# 6️ Images sans annotations
print("\nQuelques images sans annotations :")
print(images_without_annotations(images_df, annotations_df).head())
