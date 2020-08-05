import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
from src.data_process.dataset_build import generate_sample


cwd = os.getcwd()
# Images folders path's
IMAGES_FOLDER_PATH = pjoin(cwd, "data/processed")
TRAIN_FOLDER_PATH = pjoin(IMAGES_FOLDER_PATH, "train")
TEST_FOLDER_PATH = pjoin(IMAGES_FOLDER_PATH, "test")
# CSV dataset path's
CSV_DATA_PATH = pjoin(cwd, "data/datasets")
TRAIN_CSV_PATH = pjoin(CSV_DATA_PATH, "train.csv")
TEST_CSV_PATH = pjoin(CSV_DATA_PATH, "test.csv")

def create_samples(images_folder_path : str, csv_df : pd.DataFrame) -> np.ndarray:
    files_in_folder = [f for f in os.listdir(images_folder_path) if os.path.isfile(pjoin(images_folder_path, f))]
    numpy_images = []
    labels = []
    for img_file in files_in_folder:
        img_path = pjoin(images_folder_path, img_file)
        np_img, label = generate_sample(img_path, csv_df)
        numpy_images.append(np_img)
        labels.append(label)
    return np.array(numpy_images), np.array(labels)

def main():
    df = pd.read_csv(TRAIN_CSV_PATH)
    imgs, labels = create_samples(TRAIN_FOLDER_PATH, df)
    print(imgs.shape)
    print(labels.shape)

if __name__ == "__main__":
    main()