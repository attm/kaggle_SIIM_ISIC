import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import PIL
from src.data_process.data_preprocess import load_image, image_to_numpy


def get_label_from_csv(img_name : str, csv_df : pd.DataFrame) -> int:
    """
    Finds and returns label for given image name from pandas df.
    Image name column must be "image_name", label column must be "target".

    Parameters:
        img_name (str) : name of the image.
        csv_df (pd.DataFrame) : dataframe with images names and labels.
    """
    df_row = csv_df.loc[csv_df['image_name'] == img_name]
    label = df_row['target'].values[0]
    if label is not None:
        return label
    else:
        raise IndexError("get_label_from_csv: label not found, image name given is {0}".format(img_name))

def get_images_names_with_label(csv_df : pd.DataFrame, label : int) -> list:
    """
    Finds all images names with given label.
    Image name column must be "image_name", label column must be "target".

    Parameters:
        csv_df (pd.DataFrame) : dataframe with images names and labels.
        label (int) : nedeed label.
    Returns:
        img_names (list(str)) : list of images names.
    """
    df_row = csv_df.loc[csv_df['target'] == label]
    img_names = df_row['image_name'].values
    return img_names

def generate_sample(image_path : str, csv_df : pd.DataFrame) -> np.ndarray:
    """
    Generates X and y for given image in folder.
    Finds label for that image in given pd.DataFrame.

    Parameters:
        image_path (str) : Path for image file.
        csv_df (pd.DataFrame) : dataframe with images names and labels.
    Returns:
        X (np.ndarray) : image in np.ndarray format.
        y (np.ndarray) : label in np.ndarray format.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("generate_sample: can't load image, image path given is {0}".format(image_path))
    # Loading image from folder
    loaded_image = load_image(image_path)
    # Getting image name without extention
    _, image_name = os.path.split(image_path)
    image_name = os.path.splitext(image_name)[0]
    # Making numpy array from image
    numpy_image = image_to_numpy(loaded_image)
    # Getting label 
    label = get_label_from_csv(image_name, csv_df)
    return numpy_image, label
