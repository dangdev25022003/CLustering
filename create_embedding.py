# Create an instance of the Img2VecResnet18 model
import cv2
from model_lib.utils import logger
from model_lib.utility import get_config
from model_lib.utils.get_image_list import get_image_list
from model_lib.embedding import Embedding
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import os
import json

CROPS_PATH = "/mnt/sdb1/data_donghonuoc/2024/06_img"
# Create an empty dictionary to store the image feature vectors
allVectors = {}
embedding = Embedding.getInstance()


# Print a message indicating the conversion process is starting
print("Converting images to feature vectors:")

list_imgs = [f for f in os.listdir(CROPS_PATH) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]

# Iterate over each image file in the specified directory

for image in tqdm(list_imgs):
    # Open the image file
    # I = Image.open(image)
    try:
        img = cv2.imread(CROPS_PATH+'/'+image)
        # Get the feature vector representation of the image using img2vec.getVec()
        # vec = img2vec.getVec(I)
        if img is None:
            print(1)
            continue
        vec = embedding([img], [" "])
        # Store the feature vector in the allVectors dictionary, with the image filename as the key
        allVectors[image] = vec[0]
        # allVectors[image] = 0

        # Close the image file to free up system resources
        # I.close()
    except:
        continue

    # break


# Add a new image not belonging to dataset
# Open the image 'cocacola.jpeg'
# I = Image.open(IMG_PATH)
# Get the vector representation of the image using img2vec.getVec()
# vec = img2vec.getVec(I)
# Store the vector representation in a dictionary
# allVectors["coca_cola"] = vec
# Close the image file
# I.close() 

embeddings = np.array(list(allVectors.values()))
np.save('/mnt/sdb1/data_donghonuoc/clustering/embeddings/embeddings_images_meters_t6.npy', embeddings)
with open('embeddings_images_meters_t6.json', 'w') as f:
    json.dump(allVectors, f, indent=4)