import os 
import PIL
from os.path import join as pjoin
from src.data_process.data_preprocess import load_image, save_image, resize_image


cwd = os.getcwd()
RAW_DATA_FOLDER_NAME = "data/raw/small_part"
PROCESSED_DATA_FOLDER_NAME = "data/processed"
TRAIN_DATA_FOLDER_NAME = "train"
TEST_DATA_FOLDER_NAME = "test"

def process_images_from_folder(images_folder_path : str, processed_images_folder_path : str) -> None:
    """
    Processes images from given folder.

    Parameters:
        images_folder_path (str) : Folder with images that needs to be processed.
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
    img = resize_image(img)
    return img

def main():
    process_images_from_folder(pjoin(cwd, RAW_DATA_FOLDER_NAME, TRAIN_DATA_FOLDER_NAME), 
                               pjoin(cwd, PROCESSED_DATA_FOLDER_NAME, TRAIN_DATA_FOLDER_NAME))

if __name__ == "__main__":
    main()