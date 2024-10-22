import torch
import kornia
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from time import time


def randflow(img,angle=7,trans=0.07,ratio=1,sigma=15,base=500):
    h,w=img.shape[2],img.shape[3]
    # affine
    if not base is None:
        base_scale = base/torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        angle = max(w,h)/base*angle
        #print('hello')
    else:
        base_scale = 1
    rand_angle = (torch.rand(1)*2-1)*angle
    rand_trans = (torch.rand(1,2)*2-1)*trans
    M = KGT.get_affine_matrix2d(translations=rand_trans,center=torch.zeros(1,2),scale=torch.ones(1,2),angle=rand_angle)

    M = M.inverse()

    grid = KU.create_meshgrid(h,w).to(img.device)
    warp_grid = kornia.geometry.linalg.transform_points(M,grid)
    # warp_grid = grid
    #elastic
    disp = torch.rand([1,2,h,w])*2-1
    #disp = KF.gaussian_blur2d(disp,kernel_size=[(3*sigma)//2*2+1,(3*sigma)//2*2+1],sigma=[sigma,sigma])
    for i in range(5):
        disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma))
    disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma)).permute(0,2,3,1)*ratio

    disp = (disp+warp_grid-grid)*base_scale
    trans_grid = grid+disp
    mask = trans_grid<-1
    mask = torch.logical_or(trans_grid>1,mask)
    return trans_grid,trans_grid-grid,mask



def randflow_fixed(img,angle=7,trans=0.07,ratio=1,sigma=15,base=500):
    h,w=img.shape[2],img.shape[3]
    # affine
    if not base is None:
        base_scale = base/torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        angle = max(w,h)/base*angle
        #print('hello')
    else:
        base_scale = 1
    rand_angle = (torch.ones(1))*angle
    rand_trans = (torch.ones(1))*trans
    M = KGT.get_affine_matrix2d(translations=rand_trans,center=torch.zeros(1,2),scale=torch.ones(1,2),angle=rand_angle)

    M = M.inverse()

    grid = KU.create_meshgrid(h,w).to(img.device)
    warp_grid = kornia.geometry.linalg.transform_points(M,grid)
    # warp_grid = grid
    #elastic
    disp = torch.rand([1,2,h,w])*2-1
    #disp = KF.gaussian_blur2d(disp,kernel_size=[(3*sigma)//2*2+1,(3*sigma)//2*2+1],sigma=[sigma,sigma])
    for i in range(5):
        disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma))
    disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma)).permute(0,2,3,1)*ratio

    disp = (disp+warp_grid-grid)*base_scale
    trans_grid = grid+disp
    mask = trans_grid<-1
    mask = torch.logical_or(trans_grid>1,mask)
    return trans_grid,trans_grid-grid,mask




def randrot(img):
    mode = np.random.randint(0,4)
    return rot(img,mode)

def randfilp(img):
    mode = np.random.randint(0,3)
    return flip(img,mode)

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img

def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out


def checkboard(im1, im2, d=80):
    mask = torch.zeros_like(im1)
    for i in range(mask.shape[-2] // d + 1):
        for j in range(mask.shape[-1] // d + 1):
            if (i + j) % 2 == 0:
                mask[..., i * d:(i + 1) * d, j * d:(j + 1) * d] = 1
    return im1 * mask + im2 * (1 - mask)