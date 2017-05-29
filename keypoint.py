from __future__ import division

import numpy as np
import cv2
import math
import operator
import argparse
import os

class Feature(object):

	def __init__(self, left, top, right, bottom, weight):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.weight = weight

	def get_coords(self):
		return (self.left, self.top, self.right, self.bottom)

	def get_weight(self):
		return self.weight

class ROI(object):

	PARAMS = {
	"min_neighbors": 5,
	"size": 1200,
	"nfeatures": 25,
	"face_exponent":1,
	"face_threshold":0.1,
	}

	features = []

	def __init__(self, image, detectortype="ORB"):
		self.scale = 1.0
		self.processed_image = self.process_image(image, self.PARAMS["size"])
		self.DETECTOR = self.create_detector(detectortype) # ORB or AKAZE detector
		self.get_keypoint_features() # appends detector features into features list
		self.get_face_features() 	 # appends face features into features list
		self.centroid = self.get_centroid() # derives the centroid from feature placement, using weights

	# processes the image before finding ROIs to reduce cost
	def process_image(self, image, maxsize):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		h, w = image.shape[:2]

		# image already respects max bound
		if max(h, w) <= maxsize:
			return image

		# height and weight to respect max bound
		if h > w:
			self.scale = h / maxsize
			h = maxsize
			w = int(w / self.scale)
		else:
			self.scale = w / maxsize
			w = maxsize
			h = int(h / self.scale)

		return cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

	def create_detector(self, detectortype):

		if detectortype == "ORB":	
			return cv2.ORB_create(
				nfeatures=self.PARAMS["nfeatures"],      # number of features to detect
				scaleFactor=1.3,							
				patchSize=self.PARAMS["size"] // 10,
				edgeThreshold=self.PARAMS["size"] // 10,
				scoreType=cv2.ORB_HARRIS_SCORE
			)

			
		elif detectortype == "AKAZE":
			return cv2.AKAZE_create(
				threshold=0.003,		# default 0.001
				nOctaves=4,				# default 4
				nOctaveLayers=4 		# default 4
			)

	def get_keypoint_features(self):
		kps = self.DETECTOR.detect(self.processed_image)

		for kp in kps:
			ft = self.to_feature(kp=kp)
			self.features.append(ft)

	def get_face_features(self):
		cascades = ["haarcascade_frontalface_alt.xml", "haarcascade_profileface.xml", "haarcascade_eye.xml"]

		for cascade in cascades:
			c = cv2.CascadeClassifier(cascade)

			faces = c.detectMultiScale(
				self.processed_image,
				scaleFactor = 1.15,
				minNeighbors=self.PARAMS["min_neighbors"],
				minSize=(28,28),
				)

			for face in faces:
				ft = self.to_feature(face=face)
				self.features.append(ft)

	def to_feature(self, kp=None, face=None, padding=1.0):

		if kp is not None:
			x, y = kp.pt
			radius = kp.size // 2
			weight = radius ** 2 * kp.response

			return Feature(
				int((x - radius) * padding), # left
				int((y - radius) * padding), # top
				int((x + radius) * padding), # right
				int((y + radius) * padding), # bottom
				weight						 # weight
			)

		elif face is not None:
			x, y, w, h = face[:4]
			weight = w

			return Feature(
				int(x * padding), # left
				int(y * padding), # top
				int((x + w) * padding), # right
				int((y + h) * padding), # bottom
				weight						 # weight
			)

	def get_centroid(self):
		features = self.features

		sum_x = sum_y = sum_w = 0

		for feature in features:
			l, t, r, b = feature.get_coords()
			x = (l + r) // 2 # x coord
			y = (t + b) // 2 # y coords

			print feature.get_weight()

			sum_x = sum_x + x * feature.get_weight()
			sum_y = sum_y + y * feature.get_weight()
			sum_w = sum_w + feature.get_weight()

		centroid = [int(round(sum_x / sum_w)), int(round(sum_y / sum_w))]

		print centroid

		return centroid

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True)
	ap.add_argument("-d", "--detector", default="ORB")
	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])

	h, w = image.shape[:2]

	ROIS = ROI(image, args["detector"])

	for roi in ROIS.features:
		l, t, r, b = roi.get_coords() # 4-tuple coordinates of a rectangle

		scale_l = max(0, int(l * ROIS.scale))
		scale_t = max(0, int(t * ROIS.scale))
		scale_r = min(w, int(r * ROIS.scale))
		scale_b = min(h, int(b * ROIS.scale))

		cv2.rectangle(image, (scale_l, scale_t), (scale_r, scale_b), (255, 255, 255), 1)

	path = args["image"].split(".")[0] + "NEW_" + args["detector"]
	index = 1

	while(os.path.isfile(path + ".JPG")):
		path = args["image"].split(".")[0] + "NEW_" + args["detector"] + "_" + str(index)
		index = index + 1

	path = path + ".JPG"

	cv2.imwrite(path, image)

	cv2.imshow("window", image)

	cv2.waitKey(0)


if __name__ == "__main__":
	main()









