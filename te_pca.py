import os
import torch
import torch.nn as nn
# from torchvision import transforms
import torchvision
from dataset import Dataset
from architectures.architecture_pix2pix import UNet_G, ResNet_G_256x256, PatchGan_D_70x70, PatchGan_D_286x286
from architectures.architecture_pix2pixhd import Pix2PixHD_G
from trainers_advanced.trainer import Trainer
from trainers_hd_advanced.trainer_hd import Trainer_HD
from utils import save, load
from PIL import Image
from torch import optim
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, lab_to_rgb, get_sample_images_list,get_sample_images_list1
torch.manual_seed(3)#28
torch.cuda.manual_seed(3)
torch.cuda.manual_seed_all(3)
# np.random.seed(3)  # Numpy module.
# random.seed(3)  # Python random module.
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
            imgs.append((words[0], words[1], int(words[2])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # random.shuffle(imgs)
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            ])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, image, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        path = feature
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        image = Image.open(image).convert('RGB')
        # feature = cv2.imread(feature)
        if self.is_train:
            feature = self.train_tsf(feature)
            image = self.train_tsf(image)

        else:
            feature = self.test_tsf(feature)
            image = self.test_tsf(image)

        return feature, image, label, path

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


lr_D, lr_G, bs = 0.0001, 0.0001, 16
sz, ic, oc, use_sigmoid = 256, 3, 3, False
norm_type = 'instancenorm'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
# netG = UNet_G(ic, oc, sz, True, norm_type, dropout_num = 3).to(device)
netG = Pix2PixHD_G(ic, oc).to(device)

train_k = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/class_gt/gt_total.txt'
train_data = MyDataset(is_train=True, root=train_k)
val_k = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/gt_test.txt'
val_data = MyDataset(is_train=False, root=val_k)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=True)
val_loader=list(val_loader)[0]
# trainer = Trainer_HD('QPGAN', netD, netG, device, train_loader, val_loader, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
# trainer.train(100)
# trainer.train([5, 5, 5])
stage=0
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0, 0.9))
load('saved/cur_state.state', netD, netG, optimizerD, optimizerG)
sample_images_list = get_sample_images_list('Pix2pixHD_Normal', (val_loader, netG, stage, device))
plot_fig = plot_multiple_images(sample_images_list, 1, 1)
# dir = os.path.join(r'./result/test') + '/' + path[0].split('/')[-1]
save_fig('./result/test/1.jpg', plot_fig)
print("1")