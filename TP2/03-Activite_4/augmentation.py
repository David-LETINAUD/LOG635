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

#data_dir = Path.cwd() / "./EnsembleA_A2019"
data_dir = Path("./data/Markers")

###### 1. sous-ensemble de 8 classes ######
LABELS = [ "Circle2", "Circle5", "Diamond2","Diamond5", "Hexagon2", "Hexagon5", "Triangle2", "Triangle5"]


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

# Image resize
WIDTH = 28
HEIGHT = 28

# Dataset: this list will contain all the images
data_set = []


for first_level in data_dir.glob('*'):
    if first_level.is_dir():
        #print(first_level)
        data_dir_sec = Path(first_level)
        for second_level in data_dir_sec.glob('*'):
            if second_level.is_dir():
                str_lvl = str(second_level).split("\\")

                # Attention tester si label est bien dans LABELS
                label = str_lvl[len(str_lvl)-1]
                try :
                    class_num = LABELS.index(label)
                    data_dir_img = Path(second_level)

                    ###### 3. Augmentation du dataset ######
                    # On pourrait prendre en compte la quantité déjà existante pour etre sûr que le dataset est balancé                    
                    dataset_augmentation(data_dir_img,500)
                    for img_path in data_dir_img.glob('*.jpg'):
                        #   print(img_path)
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                        ###### 2. Réduction de la taille des images ######
                        resized = cv2.resize(img, (WIDTH, HEIGHT))
                        # Plot image
                        # plt.imshow(resized)
                        # plt.show(block=True)

                        data_set.append([resized, class_num, label])
                except:
                    print("Not in the dataset" + label)
          

###### 4. Création du dataset ######
# Shuffles the images
random.shuffle(data_set)

# features vector
X = []

# labels vector
y = []

# Taking features and labels from dataset
for features, class_num, label in data_set:
    print(label)
    X.append(features)
    y.append(label)

# Converts each image matrix to an image vector
#X = np.array(X).reshape(-1, WIDTH, HEIGHT, 1)

# Creating the files containing all the information about your model and saving them to the disk
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# Image 42 dimensions
X[42].shape

# Image 42 label
y[42]

# Reshaping image 42 from vector to matrix
#im = X[42].reshape(HEIGHT, WIDTH)
im = X[42]
plt.imshow(im, cmap='gray')
plt.show(block=True)

# file = open(path)
# y= pickle.load(file)
# print(y)

