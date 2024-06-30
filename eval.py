import torch
# from IPython import display
# from google.colab import files
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

from PIL import Image
from tkinter import filedialog
import tkinter as tk
import os
import numpy as np
import cv2
from shapely.geometry import Polygon
import pandas as pd
# import Path
import tqdm
from datetime import datetime

SHOW_FULLSIZE = False  # @param {type:"boolean"}
PREPROCESSING_METHOD = "none"  # @param ["stub", "none"]
SEGMENTATION_NETWORK = "tracer_b7"  # @param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
POSTPROCESSING_METHOD = "fba"  # @param ["fba", "none"]
SEGMENTATION_MASK_SIZE = 640  # @param ["640", "320"] {type:"raw", allow-input: true}
TRIMAP_DILATION = 30  # @param {type:"integer"}
TRIMAP_EROSION = 5  # @param {type:"integer"}
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


def select_images(file_paths=None) -> list:
    # Ask the user to select multiple image files
    # if file_paths is None:
    #     file_paths = [None]

    if file_paths is None:
        # Create a Tkinter root window (it won't be shown)
        root = tk.Tk()
        root.withdraw()
        file_paths = list(filedialog.askopenfilenames(title="Select image files"))
        root.destroy()

    # check if the user selected image files
    if not file_paths:
        print("No files selected")
        return [None]

    for file_path in file_paths:
        # Check if the file is an image
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print(f"File {file_path} is not an image")
            return [None]
    return file_paths


def preprocess_images(file_paths: list = None, resolution=640):

    if file_paths is None:
        return [None]
    original_images = []
    resized_images = []

    for file_path in file_paths:
        # Open the image using PIL
        original_image = Image.open(file_path)
        original_images.append(original_image)

        # Resize the image to 640x640 pixels
        resized_image = original_image.resize((resolution, resolution))
        resized_images.append(resized_image)

    return original_images, resized_images, file_paths


def extract_masks(alphaimages):
    '''
    Extracts the alpha channel from the images
    :param images:
    :return:
    '''
    masks = []
    images = []
    for img in alphaimages:
        r, g, b, a = img.split()
        masks.append(a)
        # append image without alpha channel to list
        images.append(Image.merge("RGB", (r, g, b)))
    return images, masks


if __name__ == "__main__":
    from carvekit.ml.files.models_loc import download_all

    download_all()

    config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                      preprocessing_method=PREPROCESSING_METHOD,
                      postprocessing_method=POSTPROCESSING_METHOD,
                      seg_mask_size=SEGMENTATION_MASK_SIZE,
                      trimap_dilation=TRIMAP_DILATION,
                      trimap_erosion=TRIMAP_EROSION,
                      device=DEVICE)

    interface = init_interface(config)

    original_images, resized_images, original_path = preprocess_images(file_paths=select_images(),
                                                                       resolution=SEGMENTATION_MASK_SIZE)


    # Drop images with four channels
    np_resized_images = [np.array(img)[:, :, :3] for img in resized_images]

    # ********************** Select images to process ********************** #
    root_dir = os.path.join('/', *original_path[0].split(os.sep)[:-1])
    images = interface(resized_images)


    images, masks = extract_masks(images)

    df = pd.DataFrame(columns=['image_path', 'max_area'])
    for mask, img, path0 in zip(masks, np_resized_images,original_path):
        mask_arr = np.array(mask)
        mask_arr[mask_arr > 0] = 255
        mask_rgb = np.stack((mask_arr,) * 3, axis=-1)
        img_arr = np.array(img)


        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_co = np.array([[[0, 0]]], dtype=np.int32)
        for ix, contour in enumerate(contours):
            if len(contour) > len(max_co):
                max_co = contour

        poly = Polygon(max_co.reshape(-1, 2))
        df = df.append({'image_path':path0, 'max_area':poly.area}, ignore_index=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_csv_path = f"{root_dir}/BioAreaData_{timestamp}.csv"
    df.to_csv(file_csv_path, index=False)

    # i have resized_images and masks, i want to higliht the masks on the original images
    # mask is a list of PIL images with alpha channel
    # images is a list of PIL images without alpha channel

    # for mask, img in zip(masks, np_resized_images):
    #     mask_arr = np.array(mask)
    #     mask_arr[mask_arr > 0] = 255
    #     mask_rgb = np.stack((mask_arr,) * 3, axis=-1)
    #     img_arr = np.array(img)

    # ********************** Save images ********************** #
    for ix,(img, path0) in enumerate(zip(images, original_path)):
        img.save(os.path.join(root_dir, 'carved_' + ",".join(os.path.basename(path0).split('.')[:-1]) + '.png'))

    # ********************** Display images ********************** #
    for im in enumerate(images):
        if not SHOW_FULLSIZE:
            print(f"size of images: {im[1].size}")
            im[1].show()
