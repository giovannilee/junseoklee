import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

dir_name = 'ran_test/undercut_90'
lendir = len(os.listdir('{}'.format(dir_name)))

for j in range(0, 15):
    files = os.listdir(dir_name)
    for i in range(0, lendir):
        BON = np.random.randint(2)
        range_x = np.random.randint(-300, 300)
        range_y = np.random.randint(-300, 300)

        img = cv2.imread('ran_test/undercut_90/{}.png'.format(i)) #+files[i])

        h, w = img.shape[:2]

        shift = np.float32([[1, 0, range_x], [0, 1, range_y]])

        img2 = cv2.warpAffine(img, shift, (w, h))
        #shift한 경우 배경이 검은색 범위만큼 흰색만들기
        if range_x >= 0 and range_y >= 0:
            img_mask = np.zeros((h, w))
            img_mask[:range_y, :] = 255
            img_mask[:, :range_x] = 255
            img_mask = img_mask.astype(np.uint8)
            for k in range(3):
                img2[:, :, k] += img_mask

        if range_x >= 0 and range_y < 0:
            img_mask = np.zeros((h, w))
            img_mask[range_y:, :] = 255
            img_mask[:, :range_x] = 255
            img_mask = img_mask.astype(np.uint8)
            for k in range(3):
                img2[:, :, k] += img_mask

        if range_x < 0 and range_y >= 0 :
            img_mask = np.zeros((h, w))
            img_mask[:range_y, :] = 255
            img_mask[:, range_x:] = 255
            img_mask = img_mask.astype(np.uint8)
            for k in range(3):
                img2[:, :, k] += img_mask

        if range_x < 0 and range_y < 0 :
            img_mask = np.zeros((h,w))
            img_mask[range_y:, :] = 255
            img_mask[:, range_x:] = 255
            img_mask = img_mask.astype(np.uint8)
            for k in range(3):
                img2[:, :, k] += img_mask
        #img_mask = np.zeros((h, w))
        #img_mask[:range_y, :] = 255
        #img_mask[:, :range_x] = 255
        #img_mask = img_mask.astype(np.uint8)
        #for k in range(3):
        #   img2[:, :, k] += img_mask

        # for k in range(3):
        #     arr_tmp = np.mean(img2, axis=2)
        #     mask = arr_tmp
        #     mask[mask >= 50] = 0
        #     mask[mask < 50] = 255
        #     mask = mask.astype(np.uint8)
        #
        #     img2[:, :, k] += mask

        kernel = np.ones((5, 5), np.float32) / 25
        if(BON == 1):
            img2 = cv2.filter2D(img2, -1, kernel)
        cv2.imwrite('shift_test/undercut_90/{}.png'.format(i + 73*j), img2)

        #cv2.imshow('shift image', img2)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
