import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import PIL
from src.data_process.data_preprocess import load_image


def get_label_from_csv(img_name : str, csv_df : pd.DataFrame) -> int:
    df_row = csv_df.loc[csv_df['image_name'] == img_name]
    label = df_row['target'].values[0]
    if label is not None:
        return label
    else:
        raise IndexError("get_label_from_csv: label not found, image name given is {0}".format(img_name))

def get_images_names_with_label(csv_df : pd.DataFrame, label : int) -> list:
    df_row = csv_df.loc[csv_df['target'] == label]
    img_names = df_row['image_name'].values
    return img_names

def image_to_numpy(img : PIL.Image.Image) -> np.ndarray:
    if not isinstance(img, PIL.Image.Image):
        raise TypeError("image_to_numpy: expected img of type PIL.Image.Image, got {0}".format(type(img)))
    img = img.convert("RGB")
    np_img = np.array(img)
    return np_img

def generate_sample(image_path : str, csv_df : pd.DataFrame) -> np.ndarray:
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
