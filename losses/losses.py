import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
def get_label(bs, device):
	label_r = torch.full((bs, ), 1, device = device)
	label_f = torch.full((bs, ), 0, device = device)
	return label_r, label_f

class SGAN(nn.Module):
	def __init__(self, device):
		super(SGAN, self).__init__()
		self.criterion = nn.BCELoss()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs, self.device)
		# label_r = torch.autograd.Variable(label_r)
		# label_f = torch.autograd.Variable(label_f)
		return self.criterion(c_xr, label_r.float()) + self.criterion(c_xf, label_f.float())
	# def g_loss(self, c_xf):
	# 	bs = c_xf.shape[0]
	# 	label_r, _ = get_label(bs, self.device)
	# 	return self.criterion(c_xf, label_r)
	def g_loss(self, c_xr, c_xf):
		return torch.mean(c_xr) - torch.mean(c_xf)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class LSGAN(nn.Module):
	def __init__(self, device):
		super(LSGAN, self).__init__()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs, self.device)
		return torch.mean((c_xr - label_r) ** 2) + torch.mean((c_xf - label_f) ** 2)

	def g_loss(self, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs, self.device)
		return torch.mean((c_xf - label_r) ** 2)

class HINGEGAN(nn.Module):
	def __init__(self, device):
		super(HINGEGAN, self).__init__()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		return torch.mean(torch.nn.ReLU()(1-c_xr)) + torch.mean(torch.nn.ReLU()(1+c_xf))

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class WGAN(nn.Module):
	def __init__(self, device):
		super(WGAN, self).__init__()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		return -torch.mean(c_xr) + torch.mean(c_xf)

	def g_loss(self, c_xf):
		return -torch.mean(c_xf)

class RASGAN(nn.Module):
	def __init__(self, device):
		super(RASGAN, self).__init__()
		self.device = device
		self.criterion = nn.BCEWithLogitsLoss()

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs, self.device)
		return (self.criterion(c_xr - torch.mean(c_xf), label_r) + self.criterion(c_xf - torch.mean(c_xr), label_f)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, label_f = get_label(bs, self.device)
		return (self.criterion(c_xr - torch.mean(c_xf), label_f) + self.criterion(c_xf - torch.mean(c_xr), label_r)) / 2.0

class RALSGAN(nn.Module):
	def __init__(self, device):
		super(RALSGAN, self).__init__()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs, self.device)
		return (torch.mean((c_xr - torch.mean(c_xf) - label_r)**2) + torch.mean((c_xf - torch.mean(c_xr) + label_r)**2)) / 2.0

	def g_loss(self, c_xr, c_xf):
		bs = c_xf.shape[0]
		label_r, _ = get_label(bs, self.device)
		return (torch.mean((c_xf - torch.mean(c_xr) - label_r)**2) + torch.mean((c_xr - torch.mean(c_xf) + label_r)**2)) / 2.0

class RAHINGEGAN(nn.Module):
	def __init__(self, device):
		super(RAHINGEGAN, self).__init__()
		self.device = device

	def d_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xr-torch.mean(c_xf)))) + torch.mean(torch.nn.ReLU()(1+(c_xf-torch.mean(c_xr))))) / 2.0

	def g_loss(self, c_xr, c_xf):
		return (torch.mean(torch.nn.ReLU()(1-(c_xf-torch.mean(c_xr)))) + torch.mean(torch.nn.ReLU()(1+(c_xr-torch.mean(c_xf))))) / 2.0

class QPGAN(nn.Module):
	def __init__(self, device, norm_type = 'L1'):
		super(QPGAN, self).__init__()
		self.device = device
		self.norm_type = norm_type

	def d_loss(self, c_xr, c_xf, real_images, fake_images):
		if(self.norm_type == 'L1'):
			denominator = (real_images - fake_images).abs().mean() * 10 * 2 #abs()绝对值 mean()平均值
		if(self.norm_type == 'L2'):
			denominator = (real_images - fake_images).mean().sqrt() * 10 * 2

		errD_1 = torch.mean(c_xr) - torch.mean(c_xf)
		errD_2 = (errD_1 ** 2) / denominator
		return errD_2 - errD_1

	def g_loss(self, c_xr, c_xf):
		return torch.mean(c_xr) - torch.mean(c_xf)