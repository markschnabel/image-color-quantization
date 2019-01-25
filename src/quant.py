from sklearn.cluster import MiniBatchKMeans     # MiniBatchKMeans used for speed
import numpy as np 
import argparse 
import cv2 

# Parse args
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
ap.add_argument('-c', '--clusters', required=True, type=int, 
    help='Number of clusters. This is equal to the number of colors the final output will have.')
args = vars(ap.parse_args())

IMAGE_PATH = args["image"]
CLUSTERS = args["clusters"]

# Read in image
try:
    image = cv2.imread(IMAGE_PATH)
except:
    print('Could not read image at location:', IMAGE_PATH)

# Extract width & height of image
(HEIGHT, WIDTH) = image.shape[:2]

# Convert image to L, A, B color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Reshape the image to a feature vector
image = image.reshape((image.shape[0] * image.shape[1], 3))

# Apply MiniBatchKMeans and then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = args["clusters"])
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

# reshape the feature vectors to images
quant = quant.reshape((HEIGHT, WIDTH, 3))
image = image.reshape((HEIGHT, WIDTH, 3))

# convert from L, A, B to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Quantized Image", quant)

cv2.waitKey(0)
print("Program successfully terminated")