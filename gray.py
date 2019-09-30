import cv2
import os

dir_name = 'Newdata/train'
lendir = len(os.listdir('{}/images'.format(dir_name)))

for i in range(0, lendir):
    img_color = cv2.imread('{}/images/{}.jpg'.format(dir_name,i),cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}/gray/{}.jpg'.format(dir_name,i), gray)
    #os.remove('./{}/images/{}.jpg'.format(dir_name,i))
