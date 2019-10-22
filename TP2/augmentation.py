from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
import random
import pickle
import Augmentor

# Always run this cell to display the complete output in the cells, not just the last result.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


data_dir = Path("./data/Markers")


###### 3. Augmentation du dataset ######
def dataset_augmentation(data_directory, nb_img):
    # Path to the image dataset
    p = Augmentor.Pipeline(str(data_directory), output_directory=".")

    # Operations to be performed on the images:
    # The parameter probability is used to decide if an operation is 
    # applied to an image as it is passed through the augmentation pipeline
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.75)
    p.flip_top_bottom(probability=0.75)
    p.skew_tilt(probability=0.75, magnitude=0.35)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)

    # Run the pipeline specifyin the number of images to generate
    p.sample(nb_img)


for first_level in data_dir.glob('*'):
    if first_level.is_dir():
        #print(first_level)
        data_dir_sec = Path(first_level)
        for second_level in data_dir_sec.glob('*'):
            if second_level.is_dir():
                str_lvl = str(second_level).split("\\")

                #print(second_level)
                data_dir_img = Path(second_level)
                dataset_augmentation(data_dir_img,10000)

