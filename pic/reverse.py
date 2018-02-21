import cv2
import os
import numpy as np

root=os.getcwd()
black=os.path.join(root, 'crop_no_box')
white=os.path.join(root, 'white')

image_files=os.listdir(black)

for image_file in image_files:
    image=cv2.imread(os.path.join(black,image_file))
    if len(image.shape)==3:
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=255-image
    cv2.imwrite(os.path.join(white, image_file), image)

print('finish')
