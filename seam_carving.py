from skimage import transform
from skimage import filters
import numpy
import argparse
import cv2


class SeamCarver(object):
	def __init__(self, image, ratio):
		self.image = image
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		self.ratio = ratio
		self.direction = "vertical" # default
		self.numseams = 0
		self.carved = self.seam_carve()

	def seam_carve(self):
		ratio = self.ratio
		gray = self.gray
		image = self.image

		# compute energy map
		emap = filters.sobel(gray.astype("float"))

		# shape of image
		h, w = gray.shape[:2]

		# 1:1 ratio
		if ratio == "square":
			if h >= w:
				self.direction = "horizontal"
				self.numseams = h - w
			elif w > h:
				self.direction = "vertical"
				self.numseams = w - h

		# 16:9 ratio
		elif ratio == "landscape":
			if h >= w:
				self.direction = "horizontal"
				carved_w = w
				carved_h = (carved_w * 9) // 16
				self.numseams = h - carved_h

			elif w > h:
				# if old height can contain the new height (respecting ratio)
				if (h - ((w * 9) // 16)) >= 0:
					self.direction = "horizontal"
					carved_w = w
					carved_h = (carved_w * 9) // 16
					self.numseams = h - carved_h

				else:
					self.direction = "vertical"
					carved_h = h
					carved_w = (carved_h * 16) // 9
					self.numseams = w - carved_w

		# 9:16 ratio
		elif ratio == "portrait":
			if h >= w:
				# if old width can contain the new width (respecting ratio)
				if (w - ((h * 9) // 16)) >= 0:
					self.direction = "vertical"
					carved_h = h
					carved_w = (carved_h * 9) // 16
					self.numseams = w - carved_w
				else:
					self.direction = "horizontal"
					carved_w = w
					carved_h = (carved_w * 16) // 9
					self.numseams = h - carved_h
			elif w > h:
				self.direction = "vertical"
				carved_h = h
				carved_w = (carved_h * 9) // 16
				self.numseams = w - carved_w

		carved = transform.seam_carve(image, emap, self.direction, self.numseams)

		return carved

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True)
	ap.add_argument("-r", "--ratio", default="square")
	ap.add_argument("-s", "--save", required=True)

	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])

	sc = SeamCarver(image, args["ratio"])

	carved = sc.carved

	# show original image
	#cv2.imshow("Original", image)

	print "Transformed image into {0} format, removing {1} seams along the {2}".format(args["ratio"], sc.numseams, sc.direction)

	# show carved image
	cv2.imshow("Carved", carved)
	cv2.waitKey(0)

	carved = carved * 255

	carved = carved.astype('uint8')

	cv2.imwrite(args["save"] + "/" + args["image"].split("/")[-1], carved)

if __name__ == "__main__":
	main()