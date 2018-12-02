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

from sklearn.externals import joblib
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import os
from sklearn import svm



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train", required=True, help="Path to the dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images(args["train"]):
	# extract the digit
	ishuman = imagePath.split(os.sep)[-2]

	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(500, image.shape[1]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(gray, (64, 128))
	print 'training '
	'''
	edged = imutils.auto_canny(gray)
	# find contours in the edge map, keeping only the largest one which is presumed to be the digit
	(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key=cv2.contourArea)
	# extract the digit and resize it
	(x, y, w, h) = cv2.boundingRect(c)
	img = gray[y:y + h, x:x + w]
	img = cv2.resize(img, (200, 100))
    '''
	# extract Histogram of Oriented Gradients from img
	H = feature.hog(img, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2))
	
	# update the data and labels
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
