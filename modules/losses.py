# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF
import kornia
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import kornia.utils as KU
import kornia.filters as KF
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map * mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1 > 0, img2 > 0).float()
        for i in range(self.window_size // 2):
            mask = (F.conv2d(mask, window, padding=self.window_size // 2, groups=channel) > 0.8).float()
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


ssimloss = SSIMLoss(window_size=11)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)  # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1)  # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2)  # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3)  # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4)  # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0, 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims
        sum_filt = torch.ones([1, I.shape[1], *win]).cuda() / I.shape[1]
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / ((I_var * J_var).clamp(min=1e-3) + 1e-3)
        return -1 * torch.mean(cc)


def l1loss(img1, img2, mask=1, eps=1e-2):
    mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
    mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
    mean_ = mean_.detach() / 2
    std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
    std_ = std_.detach() / 2
    img1 = (img1 - mean_) / std_
    img2 = (img2 - mean_) / std_
    img1 = KF.gaussian_blur2d(img1, (3, 3), (1, 1)) * mask_
    img2 = KF.gaussian_blur2d(img2, (3, 3), (1, 1)) * mask_
    return ((img1 - img2) * mask).abs().clamp(min=eps).mean()


def l2loss(img1, img2, mask=1, eps=1e-2):
    mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
    mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
    mean_ = mean_.detach() / 2
    std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
    std_ = std_.detach() / 2
    img1 = (img1 - mean_) / std_
    img2 = (img2 - mean_) / std_
    img1 = KF.gaussian_blur2d(img1, (3, 3), (1, 1)) * mask_
    img2 = KF.gaussian_blur2d(img2, (3, 3), (1, 1)) * mask_
    return ((img1 - img2) * mask).abs().clamp(min=eps).pow(2).mean()


class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss, self).__init__()
        self.AP5 = nn.AvgPool2d(5, stride=1, padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5, stride=1, padding=2).cuda()

    def forward(self, img1, img2, mask=1, eps=1e-2):
        # img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1 > 1e-2, img2 > 1e-2)
        mean_ = img1.mean(dim=[-1, -2], keepdim=True) + img2.mean(dim=[-1, -2], keepdim=True)
        mean_ = mean_.detach() / 2
        std_ = img1.std(dim=[-1, -2], keepdim=True) + img2.std(dim=[-1, -2], keepdim=True)
        std_ = std_.detach() / 2
        img1 = (img1 - mean_) / std_
        img2 = (img2 - mean_) / std_
        grad1 = KF.spatial_gradient(img1, order=2)
        grad2 = KF.spatial_gradient(img2, order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1 - grad2) + (grad1 - grad2).pow(2) * 10) * mask).abs().clamp(min=eps).mean()
        # l = l[...,5:-5,10:-10].mean()
        return l


def smoothloss(disp, img=None):
    smooth_d = [3 * 3, 7 * 3, 15 * 3]
    b, c, h, w = disp.shape
    grad = KF.spatial_gradient(disp, order=2).abs().sum(dim=2)[:, :, 5:-5, 5:-5].clamp(min=1e-9).mean()
    local_smooth_re = 0
    for d in smooth_d:
        local_mean = KF.gaussian_blur2d(disp, (d, d), (d // 6, d // 6), border_type='replicate')
        # local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
        local_smooth_re += 1 / (d * 1.0 + 1) * (disp - local_mean)[:, :, d // 2:-d // 2, d // 2:-d // 2].pow(2).mean()
        # local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
    # global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
    # std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
    # grad = grad[...,10:-10,10:-10]
    return 5000 * local_smooth_re + 500 * grad


def l2regularization(img):
    return img.pow(2).mean()


def orthogonal_loss(t):
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1
    c = torch.matmul(t, t)
    k = torch.linalg.eigvals(c)  # Get eigenvalues of C
    ortho_loss = torch.mean((k[0][0] - 1.0) ** 2) + torch.mean((k[0][1] - 1.0) ** 2)
    ortho_loss = ortho_loss.float()
    return ortho_loss


def determinant_loss(t):
    # Determinant Loss: determinant should be close to 1
    det_value = torch.det(t)
    det_loss = torch.sum((det_value - 1.0) ** 2) / 2
    return det_loss


def smoothness_loss(deformation, img=None, alpha=0.0):
    """Calculate the smoothness loss of the given defromation field

    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    """
    diff_1 = torch.abs(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    diff_2 = torch.abs((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    if img is not None and alpha > 0.0:
        mask = img
        weight_1 = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss


def feat_loss(feat1, feat2, grid=16):
    b, c, h, w = feat1.shape[0], feat1.shape[1], feat1.shape[2], feat1.shape[3]
    shift_x = np.random.randint(1, w // grid)
    shift_y = np.random.randint(1, h // grid)
    x = tuple(np.arange(grid) * w // grid + shift_x)
    y = tuple(np.arange(grid) * w // grid + shift_y)
    feat1_sampled = feat1[:, :, y, :]
    feat1_sampled = F.normalize(feat1_sampled[:, :, :, x], dim=1).view(b, c, -1).permute(0, 2, 1).contiguous().view(-1,
                                                                                                                    c)
    feat2_sampled = feat2[:, :, y, :]
    feat2_sampled = F.normalize(feat2_sampled[:, :, :, x], dim=1).view(b, c, -1).permute(0, 2, 1).contiguous().view(-1,
                                                                                                                    c)
    # .view(b,c,-1).permute(0,2,1).view(-1,c)
    featset = torch.cat([feat1_sampled, feat2_sampled])
    perseed = torch.randperm(featset.shape[0])
    featset = featset[perseed][0:feat1_sampled.shape[0]]
    simi_pos = (feat1_sampled * feat2_sampled).sum(dim=-1)
    simi_neg = (feat1_sampled * featset).sum(dim=-1) if torch.rand(1) > 0.5 else (feat2_sampled * featset).sum(dim=-1)
    loss = (simi_neg - simi_pos + 0.5).clamp(min=0.0).mean()
    return loss


ssim_loss = kornia.losses.SSIMLoss(11, reduction='mean')
class Fusionloss_2(nn.Module):
    def __init__(self):
        super(Fusionloss_2, self).__init__()

    def forward(self, vi, ir, fu, weights=[1, 1, 1]):
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
        grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
        loss_grad = 0.5 * F.l1_loss(grad_fus, grad_ir) + 0.5 * F.l1_loss(grad_fus, grad_vi)
        loss_ssim = 0.5 * ssim_loss(ir, fu) + 0.5 * ssim_loss(vi, fu)
        loss_intensity = 0.5 * F.l1_loss(fu, ir) + 0.5 * F.l1_loss(fu, vi)
        loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity
        return loss_intensity, loss_ssim, loss_grad, loss_total


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy().cuda()  # 计算梯度
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        # loss_in=F.l1_loss(x_in_max,generate_img)
        loss_in = 0.2 * F.l1_loss(image_ir, generate_img) + F.l1_loss(image_y, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        loss_ssim = self.ssim(image_y, generate_img) + self.ssim(image_ir, generate_img)

        loss_total = 5 * loss_in + 10 * loss_grad + 5 * loss_ssim  # 总的融合损失（不包括分割损失）

        return loss_total, loss_in, loss_grad, loss_ssim


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)  # .cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)  # .cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class grad_loss(nn.Module):
    def __init__(self):
        super(grad_loss, self).__init__()
        self.sobelconv = Sobelxy(64)

    def forward(self, image_vis, image_ir, i_B, v_B):
        vi_grad = self.sobelconv(image_vis.cpu()).cuda()
        ir_grad = self.sobelconv(image_ir.cpu()).cuda()
        i_B_grad = self.sobelconv(i_B.cpu()).cuda()
        v_B_grad = self.sobelconv(v_B.cpu()).cuda()
        grad_joint = torch.min(vi_grad, ir_grad)
        loss_grad = F.l1_loss(grad_joint, i_B_grad) + F.l1_loss(grad_joint, v_B_grad)
        return loss_grad


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
                eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()