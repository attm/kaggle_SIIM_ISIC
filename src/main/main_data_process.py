import os, shutil
import PIL
import pandas as pd
import numpy as np
from os.path import join as pjoin
from src.data_process.data_preprocess import load_image, save_image, resize_image
from src.data_process.dataset_build import get_images_names_with_label


cwd = os.getcwd()
PROCESS_TRAIN = False
PROCESS_SUBMISSION = True
# Raw data pathes for selection
RAW_DATA_FOLDER_PATH = pjoin(cwd, "data/raw/full")
TRAIN_RAW_FOLDER_PATH = pjoin(RAW_DATA_FOLDER_PATH, "train")
SUBMISSION_RAW_FOLDER_PATH = pjoin(RAW_DATA_FOLDER_PATH, "submission")
# Selected images path
SELECTED_DATA_FOLDER_PATH = pjoin(cwd, "data/raw/selected")
SELECTED_TRAIN_DATA_FOLDER_PATH = pjoin(SELECTED_DATA_FOLDER_PATH, "train")
SELECTED_SUBMISSION_DATA_FOLDER_PATH = pjoin(SELECTED_DATA_FOLDER_PATH, "submission")
# Processed images path
PROCESSED_DATA_FOLDER_PATH = pjoin(cwd, "data/processed")
PROCESSED_TRAIN_DATA_FOLDER_PATH = pjoin(PROCESSED_DATA_FOLDER_PATH, "train")
PROCESSED_SUBMISSION_DATA_FOLDER_PATH = pjoin(PROCESSED_DATA_FOLDER_PATH, "submission")
# CSV dataset path's
DATASETS_PATH = pjoin(cwd, "data/datasets")
TRAIN_CSV_PATH = pjoin(DATASETS_PATH, "train.csv")
SUBMISSION_CSV_PATH = pjoin(DATASETS_PATH, "submission.csv")
# How many samples for each label
POSITIVE_SAMPLES = 580
NEGATIVE_SAMPLES = 580

def process_images_from_folder(images_folder_path : str, processed_images_folder_path : str) -> None:
    """
    Processes images from given folder.

    Parameters:
        images_folder_path (str) : Folder with images that needs to be processed.
        processed_images_folder_path (str) : Folder where processed images will be saved.
    Returns:
        None.
    """
    if not os.path.exists(images_folder_path):
        raise FileNotFoundError("process_images_from_folder: folder not found, given folder path {0}".format(images_folder_path))

    files_in_folder = [f for f in os.listdir(images_folder_path) if os.path.isfile(pjoin(images_folder_path, f))]
    for img_file in files_in_folder:
        # Loading image
        load_img_path = pjoin(images_folder_path, img_file)
        img = load_image(load_img_path)
        # Processing image
        print("Processing image {0}".format(img_file))
        img = process_image(img)
        # Saving image
        save_img_path = pjoin(processed_images_folder_path, img_file)
        save_image(img, save_img_path)

def process_image(img : PIL.Image.Image) -> PIL.Image.Image:
    """
    Processing single image.

    Parameters:
        img (PIL.Image.Image) : Image to be proccessed.
    Returns:
        img (PIL.Image.Image) : Processed image.
    """
    img = resize_image(img)
    return img

def select_images(images_folder_path : str, selected_images_folder_path : str, images_names : np.ndarray) -> None:
    """
    Selecting images with given name from given folder.

    Parameters:
        images_folder_path (str)
        selected_images_folder_path (str)
        images_names (np.ndarray)
    Returns:
        None
    """
    files_in_folder = [f for f in os.listdir(images_folder_path) if os.path.isfile(pjoin(images_folder_path, f))]
    i = 0
    for img_file in files_in_folder:
        image_name = os.path.splitext(img_file)[0]
        if np.isin(image_name, images_names):
            img_path = pjoin(images_folder_path, img_file)
            selected_image_path = pjoin(selected_images_folder_path, img_file)
            shutil.copy2(img_path, selected_image_path)
            i += 1
    print("select_images: copied {0} images from {1} to {2}".format(i, images_folder_path, selected_images_folder_path))

def select_and_preprocess_train():
    print("Selecting and processing train data")
    # Removing and recreating selected folder
    shutil.rmtree(SELECTED_TRAIN_DATA_FOLDER_PATH)
    os.mkdir(SELECTED_TRAIN_DATA_FOLDER_PATH)

    # Removing and recreating processed folder
    shutil.rmtree(PROCESSED_TRAIN_DATA_FOLDER_PATH)
    os.mkdir(PROCESSED_TRAIN_DATA_FOLDER_PATH)

    # Selecting part of the images, placing those images into selected folder
    df = pd.read_csv(TRAIN_CSV_PATH)
    positive_images_names = get_images_names_with_label(df, 1)
    np.random.shuffle(positive_images_names)
    positive_images_names = positive_images_names[:POSITIVE_SAMPLES]
    print("Found {0} images with label 1".format(len(positive_images_names)))

    negative_images_names = get_images_names_with_label(df, 0)
    np.random.shuffle(negative_images_names)
    negative_images_names = negative_images_names[:NEGATIVE_SAMPLES]
    print("Found {0} images with label 0".format(len(negative_images_names)))

    # Selecting positive labeled images
    select_images(TRAIN_RAW_FOLDER_PATH, SELECTED_TRAIN_DATA_FOLDER_PATH, positive_images_names)
    # Selecting negative labeled images
    select_images(TRAIN_RAW_FOLDER_PATH, SELECTED_TRAIN_DATA_FOLDER_PATH, negative_images_names)

    process_images_from_folder(SELECTED_TRAIN_DATA_FOLDER_PATH, PROCESSED_TRAIN_DATA_FOLDER_PATH)

def select_and_preprocess_submission():
    print("Selecting and processing SUBMISSION data")
    shutil.rmtree(PROCESSED_SUBMISSION_DATA_FOLDER_PATH)
    os.mkdir(PROCESSED_SUBMISSION_DATA_FOLDER_PATH)

    process_images_from_folder(SUBMISSION_RAW_FOLDER_PATH, PROCESSED_SUBMISSION_DATA_FOLDER_PATH)

def main():
    if PROCESS_TRAIN:
        select_and_preprocess_train()

    if PROCESS_SUBMISSION:
        select_and_preprocess_submission()

if __name__ == "__main__":
    main()