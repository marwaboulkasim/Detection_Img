import pandas as pd
from pathlib import Path
from collections import Counter
from prepare_data.data_loader import df_images


def cntr(folder_path):
    """
    Count file extensions in a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        Counter: Counts of each file extension in the folder.
    """
    p = Path(folder_path)
    
    # Check if the folder exists
    if not p.exists():
        print(f"Folder not found: {folder_path}")
        return Counter()

    exts = [f.suffix.lower().lstrip('.') for f in p.iterdir() if f.is_file()]
    return Counter(exts)


def verify_images(df: pd.DataFrame, column_name: str, folder_path: str):
    """
    Verify the consistency between image file names in a DataFrame column 
    and the actual image files stored in a folder.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing image names.
    column_name : str
        The column in the DataFrame that holds the image file names.
    folder_path : str
        Path to the folder containing the image files.

    Returns
    -------
    tuple
        A tuple with two sets:
        - missing_in_folder: images listed in the DataFrame but not found in the folder.
        - missing_in_df: images found in the folder but not listed in the DataFrame.
    """
    p = Path(folder_path)
    
    # Check if the folder exists
    if not p.exists():
        print(f"Folder not found: {folder_path}")
        return set(), set()

    # Collect all file names in the folder
    n_names = [n.name for n in p.iterdir() if n.is_file()]
    
    # Get image names from DataFrame column
    img_names = df[column_name].to_list()
    
    # Convert both to sets for easy comparison
    set_folder = set(n_names)
    set_df = set(img_names)
    
    # Find differences
    missing_in_folder = set_df - set_folder
    missing_in_df = set_folder - set_df
    
    return missing_in_folder, missing_in_df


if __name__ == "__main__":
    # Resolve dataset folder relative to project root
    BASE_DIR = Path(__file__).resolve().parent.parent
    folder = BASE_DIR / "dataset"

    missing_in_folder, missing_in_df = verify_images(df_images, "file_name", folder)
    exts = cntr(folder)
    print(exts)

    print("Images in DataFrame but not in folder:", sorted(missing_in_folder))
    print("Images in folder but not in DataFrame:", sorted(missing_in_df))
