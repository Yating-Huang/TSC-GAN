import cv2
import os
import numpy as np

path = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/paper/ActiveModels_version7/0_mask'  # 源文件所在目录 图片文件
savefilepath = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/paper/ActiveModels_version7/0_mask/'  # 输出文件所在目录 图片文件
datanames = os.listdir(path)
for i in datanames:
    img = cv2.imread(path + '/' + str(i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = gray
    img2[:, :, 1] = gray
    img2[:, :, 2] = gray
    cv2.imwrite(savefilepath + i, img2)