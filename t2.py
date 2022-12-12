# import matplotlib.pyplot as plt # plt 用于显示图片
# import matplotlib.image as mpimg # mpimg 用于读取图片
# import numpy as np
#
# # load
# img = mpimg.imread('/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/image/9.jpg')
# # 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
# # height, width, channel=(360, 480, 3)
# h,w,c = img.shape
#
# # show
# plt.imshow(img) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()
#
# # save
# # 适用于保存任何 matplotlib 画出的图像，相当于一个 screencapture
# plt.savefig('fig_cat.jpg')

import cv2 as cv
# load
img = cv.imread('/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/image/9.jpg')
# shape=(height, width, channel)
# h,w,c = img.shape
# show
print(img.shape)
cv.imshow('window_title', img)
# save
cv.imwrite('fig_cat.jpg', img)