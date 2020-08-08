import os
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(load_path : str) -> PIL.Image.Image:
    if not os.path.exists(load_path):
        raise FileNotFoundError("load_image: file not found, given path is {0}".format(load_path))
    loaded_image = Image.open(load_path, mode="r")
    return loaded_image

def save_image(image : PIL.Image.Image, save_path : str) -> None:
    if not isinstance(image, PIL.Image.Image):
        raise TypeError("save_image: can't save image, expected image of type PIL.Image.Image, but got {0}".format(type(image)))
    image.save(save_path)

def resize_image(image : PIL.Image.Image, new_size=(512, 512)) -> PIL.Image.Image:
    resized_image = image.resize(new_size)
    return resized_image

def image_to_numpy(img : PIL.Image.Image) -> np.ndarray:
    if not isinstance(img, PIL.Image.Image):
        raise TypeError("image_to_numpy: expected img of type PIL.Image.Image, got {0}".format(type(img)))
    img = img.convert("RGB")
    np_img = np.array(img, dtype="float16")
    np_img *= 1.0/255.0
    return np_img