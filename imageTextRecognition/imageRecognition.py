import cv2
import numpy as np
import glob
from skimage.io import imread, imshow
from skimage import exposure
from skimage import feature
from skimage.feature import hog
import imutils
import matplotlib.pyplot as plt

data=[]
for fil in glob.glob("/content/drive/My Drive/INFO 7390/data/streetData/*.jpg"):
    image = cv2.imread(fil) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    imgBlur = cv2.medianBlur(gray_image,5)
    imgThresh = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)
    imgDilate = cv2.dilate(imgThresh, kernel, iterations = 1)
    imgErosion = cv2.erode(imgDilate, kernel, iterations = 1)
    imgOpening = cv2.morphologyEx(imgErosion, cv2.MORPH_OPEN, kernel)
    imgCanny = cv2.Canny(imgOpening, 100, 200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
	  # find contours in the edge map, keeping only the largest one which
	  # is presmumed to be the car logo
    cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
	  # extract the logo of the car and resize it to a canonical width
	  # and height
    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y + h, x:x + w]
    logo = cv2.resize(logo, (200, 100))
	  # extract Histogram of Oriented Gradients from the logo
    fd, hog_image = hog(logo, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualize=True, multichannel=False)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.show()
	  # update the data and labels
    #data.append(H)
	  #labels.append(make)

#hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#hogImage = hogImage.astype("uint8")
#imshow("HOG Image", hogImage)