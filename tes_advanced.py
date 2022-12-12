import os
from PIL import Image
import torch
import torchvision
import sys
# from efficientnet_pytorch import EfficientNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#import d2lzh_pytorch as d2l
from time import time
import time
import  csv
# from cla import densenet, alexnet, googlenet, mnasnet, mobilenet, resnet, shufflenetv2, squeezenet, vgg, vgg16, myclassnet,inception,PNet
import numpy as np
import random
from architectures.architecture_pix2pix import UNet_G, ResNet_G_256x256, PatchGan_D_70x70, PatchGan_D_286x286
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, lab_to_rgb, get_sample_images_list1
from architectures.architecture_pix2pixhd import Pix2PixHD_G
import matplotlib.pyplot as plt
import matplotlib.image as mp
from sklearn.metrics import confusion_matrix
from PIL import Image
from utils import save, load
from torchvision.utils import save_image
import cv2 as cv
torch.manual_seed(3)#28
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
np.random.seed(3)  # Numpy module.
random.seed(3)  # Python random module.
torch.manual_seed(3)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train,root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r', encoding="utf-8")  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # random.shuffle(imgs)
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154])
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                # torchvision.transforms.Normalize(mean=[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154])
            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        mask= self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path = mask
        feature = Image.open(mask)
        # feature = cv.imread(mask)
        # feature = lab_to_rgb(feature)
        # print(feature.shape)
        # plt.imshow(feature)
        # plt.show()
        if self.is_train:
            feature = self.train_tsf(feature)

        else:
            feature = self.test_tsf(feature)
        # print(feature.shape)
        # plt.imshow(feature)
        # plt.show()
        return feature, path

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
lr_D, lr_G, bs = 0.0002, 0.0002, 16
sz, ic, oc, use_sigmoid = 256, 3, 3, False
# sz, ic, oc, use_sigmoid = 256, 1, 3, False
norm_type = 'instancenorm'
lr_D= 0.0002
lr_G=0.0002

def k_fold(device,batch_size): #,optimizer,loss,net

    test_k = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/gt_test.txt'
    #'/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/paper/ActiveModels_version7/2_mask.txt'
    test_data = MyDataset(is_train=False, root=test_k)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
    net = UNet_G(3, oc, sz, True, norm_type, dropout_num=3).to(device)
    # net = Pix2PixHD_G(ic, oc).to(device)
    n_param = sum([np.prod(param.size()) for param in net.parameters()])
    print('Network parameters: ' + str(n_param))
    net = net.cuda()
    # net = torch.nn.DataParallel(net)
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0, 0.9))
    optimizerG = optim.Adam(net.parameters(), lr=lr_G, betas=(0, 0.9))
    # print(1)
    load('saved/cur_state_1.state', netD, net, optimizerD, optimizerG)
    tes(test_loader, net, device)
    # net.load_state_dict(torch.load('./best_model/fold_' + str(i + 1) + '_3.pth'))  # ./best_model/fold_'

def tes(test_iter, net, device):
    net = net.to(device)
    print("training on ", device)

    net.eval()
    with torch.no_grad():
        pre = []
        batch_count = 0
        time_total = 0
        for X, path in test_iter:
            start_time = time.time()
            # X = X.squeeze()
            # print(X.shape)
            dir = os.path.join(r'./result/test') + '/' + path[0].split('/')[-1]
            # mp.imsave(dir, X)
            # print(X.shape)
            X = X.to(device)
            sample_fake_images = net(X).detach().cpu().numpy()
            # sample_fake_images = sample_fake_images.squeeze()
            # sample_fake_images = np.transpose(sample_fake_images, (1, 2, 0))
            #
            # plt.imshow(sample_fake_images)
            # plt.show()
            # plt.savefig(dir)
            # plt.gca().set_axis_off()
            print(sample_fake_images.shape)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            sample_images_list = get_sample_images_list1('Pix2pix_Normal', (sample_fake_images, net, device))
            plot_fig = plot_multiple_images(sample_images_list, 1, 1)
            save_fig(dir, plot_fig)

            batch_count += 1
            print(batch_count)
            time_taken = time.time() - start_time
            time_total += time_taken

        time_aver = time_total / len(test_iter)

        return time_aver  #最好test的值,对应的trian的loss跟精度



batch_size=1
k = 5
num_epochs=2
test_time=k_fold(device,batch_size) #,optimizer,loss,net
# print('%d-fold test: test_time %.5f' % (k, test_time))
print("Congratulations!!! hou bin")


