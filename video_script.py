import cv2
import numpy as np
import glob
import os

img_path = './trpo_images/*.png'
video_path = './videos/trpo.mp4'

img_array = []
for filename in sorted(glob.glob(img_path), key=os.path.getmtime):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), 10.0, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    key = cv2.waitKey(1) 
out.release()