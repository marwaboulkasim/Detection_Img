import pandas as pd
from prepare_data.data_loader import json_loader, json_df
from prepare_data.data_explorer import verify_images, get_images_without_annotations, detect_bbox_outliers
from prepare_data.data_cleaner import delete_images, clean_coco




def loader():
    """Pipeline to load JSON and convert it to DataFrames."""
    # load the json
    json_data = json_loader("dataset/data/_annotations.coco.json")
    
    # convert json into dataframe
    df_categories, df_images, df_annotations = json_df(json_data)
    #print(df_images.head()) 
    return json_data, df_categories, df_images, df_annotations




def explorer():
    # function return the missing images in the project folder & the dataframe
    json_data, df_categories, df_images, df_annotations = loader()   
    missed_in_folder, missed_in_df = verify_images(df_images, "file_name", "dataset/data")
    # end function 
    
    # function get images without annotations
    # imgs_without_ann = get_images_without_annotations("dataset/data", "dataset/data/_annotations.coco.json")
    # end function 
    
    # function detect bbox outliers
    outliers = detect_bbox_outliers(df_annotations)
    print(f"Outlier annotations: {outliers}")
    # end function 
    

    
    return missed_in_folder, missed_in_df, outliers


def cleaner():
    # list of images without annotation
    list_imgs = get_images_without_annotations("dataset/data", "dataset/data/_annotations.coco.json")
    # functuon to delete the images without annotatio 
    delete_imgs = delete_images("dataset/data", list_imgs)
    
    # function clean coco file 
    clean_file = clean_coco("dataset/data/_annotations.coco.json", "dataset/data")
    return delete_imgs, clean_file