import pandas as pd
from prepare_data.data_loader import df_images, df_annotations, df_categories


def check_nan(df):
    """
    Check for missing values (NaN) in a DataFrame and return only the columns with at least one NaN.

    Args:
        df (pd.DataFrame): The DataFrame to check for missing values.

    Returns:
        pd.Series: A series containing only the columns with NaN values and their counts.
    """
    nan_counts = df.isna().sum()
    return nan_counts[nan_counts > 0]


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


def categories_names(df):
    """
    Return a list of category names from a categories DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the 'name' column for categories.

    Returns:
        list: List of category names.
    """
    return df['name'].tolist()


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


if __name__ == "__main__":
    # --- Check NaN values ---
    for name, df in {"Annotations": df_annotations, "Images": df_images, "Categories": df_categories}.items():
        nan_cols = check_nan(df)
        if not nan_cols.empty:
            print(f"{name} NaN columns:\n{nan_cols}\n")
        else:
            print(f"No NaN values in {name} DataFrame\n")

    # --- Total counts ---
    print(f"Total number of images: {total_number(df_images)}")
    print(f"Total number of annotations: {total_number(df_annotations)}")
    print(f"Total number of categories: {total_number(df_categories)}\n")

    # --- Category names ---
    print("Category names:")
    print(categories_names(df_categories), "\n")

    # --- Images per category ---
    print("Images per category:")
    print(images_per_category(df_annotations, df_categories), "\n")

    # --- Annotations statistics ---
    annotations_per_image, stats = annotations_statistics(df_annotations)
    print("Number of annotations per image (preview):")
    print(annotations_per_image.head(), "\n")

    print("Summary statistics of annotations per image:")
    print(stats)
