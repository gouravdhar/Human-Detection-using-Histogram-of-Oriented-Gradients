# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import os
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn import svm

def batch(start,number=1500):
	global data,labels
	return data[start:start+number],labels[start:start+number]

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train", required=True, help="Path to the dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
data = []
labels = []

(winW, winH) = (64, 128)

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



for imagePath in paths.list_images(args["train"]):

	ishuman = imagePath.split(os.sep)[-2]

	image = cv2.imread(imagePath)

	for resized in pyramid(image, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=64, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

			img = cv2.resize(gray, (64, 128))

			(H,hogimg) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
				    		cells_per_block=(3, 3), transform_sqrt=True, visualise=True)
			
			data.append(H)
			labels.append(ishuman)


 
# uncomment for using svm
# svc = svm.SVC(kernel='linear', C=1,gamma=1).fit(data, labels)
# joblib.dump(svc, 'trained_detector.pkl') 
# #save trained files

# uncomment for using sgd classifier
# cl=SGDClassifier(loss="hinge",penalty="l2")
# for i in range((len(data)/1500)+1):
# 	data_,label_=batch(i*1500)
# 	print 'k'
# 	cl.partial_fit(data_,label_,classes=np.unique(labels))

# joblib.dump(cl, 'trained_detector.pkl') 

model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
joblib.dump(model, 'trained_detector_knn.pkl') 