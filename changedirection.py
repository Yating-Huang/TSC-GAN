# -*- coding: UTF-8 -*-
# !/usr/bin/env python

from PIL import Image
import os
im_num = []
for line in open("/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/paper/paper_multi/result/kfold_1/mtl/data1/test_4.txt", "r"):
    words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
    im_num.append((words[0], words[1], words[2]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
    print(words[1])
    # im_num.append(line)
# print(im_num)
#     a=words[0]
    a = os.path.split(words[1])[-1]
    print(a)
    im = Image.open(words[1])
    # b = a.split(a)[-1]
    # print(b)
    tar_name = 'data/file/val/input/'+a
    print(tar_name)
    im.save(tar_name)  # 另存
    im.close()

# for a in im_num[0]:
#     # im_name = 'data/file/train/input/{}'.format(a[:-1]) + '.jpg'
#     print(a)
#     im = Image.open(a)  # 打开指定路径下的图像
#
#     tar_name = 'data/file/train/input/{}'.format(a[:-1]) + '.jpg'
#     print(tar_name)
#     im.save(tar_name)  # 另存
#     im.close()