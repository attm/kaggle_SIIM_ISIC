import os, shutil
from os.path import join as pjoin
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from src.data_process.dataset_build import generate_sample
from src.data_process.data_preprocess import image_to_numpy, load_image


cwd = os.getcwd()
CREATE_TRAIN = True
CREATE_SUBMISSION = True

# Images folders path's
IMAGES_FOLDER_PATH = pjoin(cwd, "data/processed")
TRAIN_FOLDER_PATH = pjoin(IMAGES_FOLDER_PATH, "train")
SUBMISSION_FOLDER_PATH = pjoin(IMAGES_FOLDER_PATH, "submission")
# CSV dataset path's
DATASETS_PATH = pjoin(cwd, "data/datasets")
TRAIN_CSV_PATH = pjoin(DATASETS_PATH, "train.csv")
SUBMISSION_CSV_PATH = pjoin(DATASETS_PATH, "submission.csv")

def create_samples(images_folder_path : str, csv_df : pd.DataFrame) -> np.ndarray:
    files_in_folder = [f for f in os.listdir(images_folder_path) if os.path.isfile(pjoin(images_folder_path, f))]
    numpy_images = []
    labels = []
    for img_file in files_in_folder:
        img_path = pjoin(images_folder_path, img_file)
        np_img, label = generate_sample(img_path, csv_df)
        numpy_images.append(np_img)
        labels.append(label)
    return np.array(numpy_images, dtype="int8"), np.array(labels, dtype="int8")

def create_submission_dataset(images_folder_path : str, csv_df : pd.DataFrame) -> np.ndarray:
    files_in_folder = [f for f in os.listdir(images_folder_path) if os.path.isfile(pjoin(images_folder_path, f))]
    numpy_images = []
    images_names_from_df = csv_df['image_name'].values
    for image_name in images_names_from_df:
        if not image_name + ".jpg" in files_in_folder:
            raise FileNotFoundError("create_submission_dataset: image named {0} presented in csv file, but NOT found in folder".format(image_name))
        
        img_path = pjoin(images_folder_path, image_name)
        img = load_image(img_path + ".jpg")
        numpy_images.append(image_to_numpy(img))
    
    return np.array(numpy_images, dtype="int8")

def main():
    if CREATE_TRAIN:
        # Creating train numpy dataset
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        train_images, train_labels = create_samples(TRAIN_FOLDER_PATH, train_df)
        np.save(pjoin(DATASETS_PATH, "X_train.npy"), train_images)
        np.save(pjoin(DATASETS_PATH, "y_train.npy"), train_labels)
        print("Created train dataset, shape of X is {0}, dtype is {1}".format(train_images.shape, train_images.dtype))
        print("First element of train dataset is {0}".format(train_images[0]))
        print("Created train labels, shape is {0}, dtype is {1}".format(train_labels.shape, train_labels.dtype))
        print("First 10 elements of train labels is {0}".format(train_labels[:10]))

    if CREATE_SUBMISSION:
        # Creating submission numpy dataset
        submission_df = pd.read_csv(SUBMISSION_CSV_PATH)
        submission_images = create_submission_dataset(SUBMISSION_FOLDER_PATH, submission_df)
        np.save(pjoin(DATASETS_PATH, "X_submission.npy"), submission_images)
        print("Created submission dataset, shape of X is {0}, dtype is {1}".format(submission_images.shape, submission_images.dtype))
        print("First element of submission dataset is {0}".format(submission_images[0]))

if __name__ == "__main__":
    main()