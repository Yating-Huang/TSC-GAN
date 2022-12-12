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
# direction='/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/paper/project/generative-pix2pix-master/data/file/'
# train_dir_name = [direction+'train_2/input', direction+'train_2/target']
# val_dir_name = [direction+'val_2/input', direction+'val_2/target']
# train_dir_name = ['data/file/train/input', 'data/file/train/target']
# val_dir_name = ['data/file/val/input', 'data/file/val/target']

lr_D, lr_G, bs = 0.0001, 0.0001, 8
sz, ic, oc, use_sigmoid = 256, 3, 3, False
norm_type = 'instancenorm'

# dt = {
# 	'input' : transforms.Compose([
# 		transforms.Resize((256, 256)),
# 		transforms.ToTensor(),
# 		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 	]),
# 	'target' : transforms.Compose([
# 		transforms.Resize((256, 256)),
# 		transforms.ToTensor(),
# 		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #[0.430, 0.317, 0.325], std=[0.163, 0.146, 0.154]
# 	])
# }

# train_data = Dataset(train_dir_name, basic_types = 'Pix2Pix', shuffle = True)
# val_data = Dataset(val_dir_name, basic_types = 'Pix2Pix', shuffle = False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PatchGan_D_70x70(ic, oc, use_sigmoid, norm_type).to(device)
# netD = TSCNet(ic, oc, use_sigmoid, norm_type).to(device)
# netG = UNet_G(ic, oc, sz, True, norm_type, dropout_num = 3).to(device)
netG = Pix2PixHD_G(ic, oc).to(device)


# trn_dl = train_data.get_loader(256, bs, data_transform = dt)
# # print(1)
# val_dl = list(val_data.get_loader(256, 3, data_transform = dt))[0]
train_k = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_2/mask_label/data2_total.txt' #/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_2/mask_label/
train_data = MyDataset(is_train=True, root=train_k)
val_k = '/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_2/mask_label/data_val.txt' #/media/yating/e7954a13-c9b5-4ef0-8768-97c22947c767/chinese_medicine/data_1/885_gt_image/gt_val.txt
val_data = MyDataset(is_train=False, root=val_k)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=bs, shuffle=True)
val_loader=list(val_loader)[0]
# trainer = Trainer_HD('QPGAN', netD, netG, device, train_loader, val_loader, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer_HD('QPGAN', netD, netG, device, train_loader, val_loader, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer.train(100)
# trainer.train([5, 5, 5])
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)