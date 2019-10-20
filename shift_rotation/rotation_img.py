
import cv2
import os
import random
import imutils

dir_name = 'new_sam_img/undercut'
lendir = len(os.listdir('{}'.format(dir_name)))

for i in range(0, lendir):
# load the image and show it
    image = cv2.imread("new_sam_img/undercut/{}.png".format(i))
    cv2.imshow("Original", image)

    # grab the dimensions of the image and calculate the center of the image
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

# rotate our image by 45 degrees
    M = cv2.getRotationMatrix2D((cX, cY), 270, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite('ran_train/undercut_270/{}.png'.format(i), rotated)

