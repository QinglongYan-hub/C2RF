import torch
import torch.nn as nn
import kornia
import numpy as np
import kornia.filters as KF
import torch.nn.functional as F
from modules.losses import *
import sys
import os
from utils.utils import RGB2YCrCb, YCbCr2RGB
from modules.RegNet import registration_net, SpatialTransformer, get_scheduler, gaussian_weights_init
from modules.FusionNet import AE_Encoder, AE_Decoder, Fusion_layer, Vgg19, VGGInfoNCE
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)



class C2RF(nn.Module):
    def __init__(self, opts=None):
        super(C2RF, self).__init__()
        # Registration Network
        self.reg_net = registration_net()
        self.resume_flag = False
        self.ST = SpatialTransformer(256, 256, True)

        # Fusion Network
        self.AE_encoder = AE_Encoder().eval()
        self.AE_decoder = AE_Decoder().eval()
        self.fusion_layer = Fusion_layer().eval()

        # Contrastive Learning
        self.VGG = Vgg19(requires_grad=False)
        self.contra_loss = VGGInfoNCE(model_vgg19=self.VGG)

        # Optimizers
        lr = 0.001
        self.reg_net_opt = torch.optim.Adam(self.reg_net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)

        # Loss
        self.gradientloss = gradientloss()
        self.ncc_loss = ncc_loss()
        self.ssim_loss = ssimloss
        self.weights_sim = [1, 1, 0.2]
        self.weights_ssim1 = [0.3, 0.7]
        self.weights_ssim2 = [0.7, 0.3]

        # Others
        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()

    def load_and_freeze(self, model, model_path, filename):
        model.load_state_dict(torch.load(os.path.join(model_path, filename)))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    
    def fusion_fixed(self, model_path):
        self.load_and_freeze(self.AE_encoder, model_path, 'Encoder.pth')
        self.load_and_freeze(self.AE_decoder, model_path, 'Decoder.pth')
        self.load_and_freeze(self.fusion_layer, model_path, 'Fusion_layer.pth')

    def initialize(self):
        self.reg_net.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.reg_net_sch = get_scheduler(self.reg_net_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.reg_net.cuda(self.gpu)
        self.AE_encoder.cuda(self.gpu)
        self.AE_decoder.cuda(self.gpu)
        self.fusion_layer.cuda(self.gpu)

    def setg_epoch(self, epoch):
        self.epoch = epoch

    def generate_mask(self):
        flow = self.ST.grid + self.disp
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        if self.border_mask.device != goodmask.device:
            self.border_mask = self.border_mask.to(goodmask.device)
        self.goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        for i in range(2):
            self.goodmask = (self.AP(self.goodmask) > 0.3).float()
        flow = self.ST.grid - self.disp
        goodmask = F.grid_sample(self.goodmask, flow)
        self.goodmask_inverse = goodmask
    
    def forward(self, vi_warp, ir, vi):
        b = vi_warp.shape[0]
        vi_warp_Y, vi_warp_Cb, vi_warp_Cr = RGB2YCrCb(vi_warp)
        ir_Y, ir_Cb, ir_Cr = RGB2YCrCb(ir)
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)

        F_i1, F_i2, F_ib, F_id, F_v1_warp, F_v2_warp, F_vb_warp, F_vd_warp = self.AE_encoder(ir_Y, vi_warp_Y)
        deformation = self.reg_net(F_ib, F_vb_warp, type='bi')  # vi'-->ir
        disp = deformation['vis2ir']
        vi_reg, flow = self.ST(vi_warp, disp)
        F_v2_reg, flow = self.ST(F_v2_warp, disp)
        F_vb_reg, flow = self.ST(F_vb_warp, disp)
        F_vd_reg, flow = self.ST(F_vd_warp, disp)

        ir_vi_B, ir_vi_D, ir_vi_cross = self.fusion_layer(F_ib, F_vb_reg, F_id, F_vd_reg, F_i2, F_v2_reg)
        fusion_reg = self.AE_decoder(ir_vi_B, ir_vi_D, ir_vi_cross)
        vi_reg_Y, vi_reg_Cb, vi_reg_Cr = RGB2YCrCb(vi_reg)
        fusion_reg_RGB = YCbCr2RGB(fusion_reg, vi_reg_Cb, vi_reg_Cr)

        return vi_reg, fusion_reg_RGB, disp, flow


    def update_RF(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp,
                  image_ir_warp_2, image_vi_warp_2, disp_2,
                  image_ir_warp_3, image_vi_warp_3, disp_3):
        (self.image_ir_RGB, self.image_vi_RGB, self.image_ir_warp_RGB, self.image_vi_warp_RGB, self.disp,
         self.image_ir_warp_RGB_2, self.image_vi_warp_RGB_2, self.disp_2,
         self.image_ir_warp_RGB_3, self.image_vi_warp_RGB_3, self.disp_3) = (
        image_ir, image_vi, image_ir_warp, image_vi_warp, disp,
        image_ir_warp_2, image_vi_warp_2, disp_2,
        image_ir_warp_3, image_vi_warp_3, disp_3)

        self.reg_net_opt.zero_grad()
        self.train_forward_RF()
        self.backward_RF()
        self.reg_net_opt.step()
    
    def extract_Y(self, image_rgb):
        Y, _, _ = RGB2YCrCb(image_rgb)
        return Y

    def train_forward_RF(self):
        b = self.image_ir_warp_RGB.shape[0]
        images = [self.image_ir_RGB, 
                  self.image_vi_RGB, 
                  self.image_vi_warp_RGB,
                  self.image_vi_warp_RGB_2, 
                  self.image_vi_warp_RGB_3]
        Y_channels = [self.extract_Y(image) for image in images]
        (self.image_ir_Y, 
         self.image_vi_Y, 
         self.image_vi_warp_Y, 
         self.image_vi_warp_Y_2, 
         self.image_vi_warp_Y_3) = Y_channels

        # Encoder
        F_i1, F_i2, F_ib, F_id, F_v1, F_v2, F_vb, F_vd = self.AE_encoder(self.image_ir_Y, self.image_vi_warp_Y)
        # Registration using common features 
        # Here, b(base)--common, d(detail)--unnique
        deformation = self.reg_net(F_ib, F_vb, type='bi')
        self.down2 = deformation['down2']
        self.down4 = deformation['down4']
        self.down8 = deformation['down8']
        self.deformation_1['ir2vis'] = deformation['ir2vis']
        self.deformation_1['vis2ir'] = deformation['vis2ir']
        img_stack = torch.cat([self.image_ir_RGB, self.image_vi_warp_RGB])
        disp_stack = torch.cat([deformation['ir2vis'], deformation['vis2ir']])
        img_warp_stack, flow_ = self.ST(img_stack, disp_stack)
        self.image_ir_warp_fake_RGB, self.image_vi_Reg_RGB = torch.split(img_warp_stack, b, dim=0)

        # Decoder & Fusion layer
        self.image_vi_Reg_Y, self.image_vi_Reg_Cb, self.image_vi_Reg_Cr = RGB2YCrCb(self.image_vi_Reg_RGB)
        ir_stack_reg_Y = torch.cat([self.image_ir_Y, 
                                    self.image_ir_Y, 
                                    self.image_ir_Y, 
                                    self.image_ir_Y, 
                                    self.image_ir_Y])
        vi_stack_reg_Y = torch.cat([self.image_vi_Reg_Y, 
                                    self.image_vi_Y, 
                                    self.image_vi_warp_Y, 
                                    self.image_vi_warp_Y_2,
                                    self.image_vi_warp_Y_3])
        F_i1, F_i2, F_ib, F_id, F_v1, F_v2, F_vb, F_vd = self.AE_encoder(ir_stack_reg_Y, vi_stack_reg_Y)
        ir_vi_B, ir_vi_D, ir_vi_cross = self.fusion_layer(F_ib, F_vb, F_id, F_vd, F_i2, F_v2)
        fusion_img = self.AE_decoder(ir_vi_B, ir_vi_D, ir_vi_cross)
        self.image_fusion_1, self.image_fusion_gt, self.image_fusion_warp, self.image_fusion_warp_2, self.image_fusion_warp_3 = torch.split(
            fusion_img, b, dim=0)

        # generate_mask
        self.generate_mask()
        self.image_display = torch.cat((self.image_ir_RGB[0:1, 0:1], self.image_ir_warp_RGB[0:1, 0:1]), dim=0).detach()

    def backward_RF(self):
        mask_ = torch.logical_and(self.image_ir_Y > 1e-5, self.image_vi_Y > 1e-5)
        mask_ = torch.logical_and(self.image_vi_Reg_Y > 1e-5, mask_)
        mask_ = mask_ * self.goodmask * self.goodmask_inverse
        loss_reg_img = self.imgloss(self.image_ir_warp_RGB, self.image_ir_warp_fake_RGB, self.goodmask) + \
                       self.imgloss(self.image_vi_Reg_RGB, self.image_vi_RGB, self.goodmask * self.goodmask_inverse) # + \
                    #    self.imgloss(self.image_fusion_1, self.image_fusion_gt, mask_)
        loss_reg_field = self.weightfiledloss(self.image_vi_warp_RGB, self.image_ir_warp_fake_RGB,
                                              self.deformation_1['ir2vis'], self.disp.permute(0, 3, 1, 2))
        loss_smooth = smoothloss(self.down2) + smoothloss(self.down4) + smoothloss(self.down8)
        loss_border_re = 0.5 * self.border_suppression(self.image_vi_Reg_RGB, self.goodmask_inverse) + \
                         self.border_suppression(self.image_ir_warp_fake_RGB, self.goodmask)
        # contrastive loss
        self.image_fusion_1_ = self.image_fusion_1.repeat(1, 3, 1, 1) * self.goodmask * self.goodmask_inverse
        self.image_fusion_gt_ = self.image_fusion_gt.repeat(1, 3, 1, 1) * self.goodmask * self.goodmask_inverse
        self.image_fusion_warp_ = self.image_fusion_warp.repeat(1, 3, 1, 1)
        self.image_fusion_warp_2_ = self.image_fusion_warp_2.repeat(1, 3, 1, 1)
        self.image_fusion_warp_3_ = self.image_fusion_warp_3.repeat(1, 3, 1, 1)
        self.image_fusion_warp_ = [self.image_fusion_warp_, self.image_fusion_warp_2_, self.image_fusion_warp_3_]
        loss_Contrastive = self.contra_loss(self.image_fusion_1_, self.image_fusion_gt_, self.image_fusion_warp_)

        loss_total = loss_reg_img * 1 + loss_reg_field * 10 + loss_smooth  + loss_border_re * 10 + loss_Contrastive * 10
        (loss_total).backward()

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field
        self.loss_smooth = loss_smooth
        self.loss_border = loss_border_re
        self.loss_contra = loss_Contrastive
        self.loss_total = loss_total

    def imgloss(self, src, tgt, mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + \
               weights[1] * self.gradientloss(src, tgt, mask)

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean()

    def fusloss(self, ir, vi, fu, weights=[1, 0, 0.5, 0]):
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
        grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
        loss_grad = 0.5 * F.l1_loss(grad_fus, grad_ir) + 0.5 * F.l1_loss(grad_fus, grad_vi)
        loss_ssim = 0.5 * self.ssim_loss(ir, fu) + 0.5 * self.ssim_loss(vi, fu)
        loss_intensity = 0.5 * F.l1_loss(fu, ir) + 0.5 * F.l1_loss(fu, vi)
        loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity
        return loss_intensity, loss_ssim, loss_grad, loss_total

    def update_lr(self):
        self.reg_net_sch.step()

    def save(self, filename, ep, total_it):
        torch.save(self.reg_net.state_dict(), filename)
        return

    def outputs_image(self):
        images_ir = self.normalize_image(self.image_ir_RGB).detach()
        images_vi = self.normalize_image(self.image_vi_RGB).detach()
        images_ir_warp = self.normalize_image(self.image_ir_warp_RGB).detach()
        images_vi_warp = self.normalize_image(self.image_vi_warp_RGB).detach()
        images_vi_Reg = self.normalize_image(self.image_vi_Reg_RGB).detach()
        images_fusion_1 = self.normalize_image(self.image_fusion_1).detach()
        images_fusion_gt = self.normalize_image(self.image_fusion_gt).detach()
        images_fusion_warp = self.normalize_image(self.image_fusion_warp).detach()
        goodmask_inverse_ = self.goodmask_inverse
        goodmask_ = self.goodmask

        return images_ir, images_vi, images_ir_warp, images_vi_warp, images_vi_Reg, \
               images_fusion_1, images_fusion_gt, images_fusion_warp, goodmask_, goodmask_inverse_

    def normalize_image(self, x):
        return x[:, 0:1, :, :]
    
    
    def update_RF_warmup(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp):
        (self.image_ir_RGB, self.image_vi_RGB, self.image_ir_warp_RGB, self.image_vi_warp_RGB, self.disp) = (
        image_ir, image_vi, image_ir_warp, image_vi_warp, disp)

        self.reg_net_opt.zero_grad()
        self.train_forward_RF_warmup()
        self.backward_RF_warmup()
        self.reg_net_opt.step()


    def train_forward_RF_warmup(self):
        b = self.image_ir_warp_RGB.shape[0]
        ir_Y, ir_Cb, ir_Cr = RGB2YCrCb(self.image_ir_RGB)
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(self.image_vi_RGB)
        vi_warp_Y, vi_warp_Cb, vi_warp_Cr = RGB2YCrCb(self.image_vi_warp_RGB)
        self.image_ir_Y = ir_Y
        self.image_vi_Y = vi_Y
        self.image_vi_warp_Y = vi_warp_Y

        # Encoder
        ir_stack = self.image_ir_Y
        vi_stack = self.image_vi_warp_Y
        F_i1, F_i2, F_ib, F_id, F_v1, F_v2, F_vb, F_vd = self.AE_encoder(ir_stack, vi_stack)
        # Registration using common features
        deformation = self.reg_net(F_ib, F_vb, type='bi')
        self.down2 = deformation['down2']
        self.down4 = deformation['down4']
        self.down8 = deformation['down8']
        self.deformation_1['ir2vis'] = deformation['ir2vis']
        self.deformation_1['vis2ir'] = deformation['vis2ir']
        img_stack = torch.cat([self.image_ir_RGB, self.image_vi_warp_RGB])
        disp_stack = torch.cat([deformation['ir2vis'], deformation['vis2ir']])
        img_warp_stack, flow_ = self.ST(img_stack, disp_stack)
        self.image_ir_warp_fake_RGB, self.image_vi_Reg_RGB = torch.split(img_warp_stack, b, dim=0)

        # Decoder & Fusion layer
        self.image_vi_Reg_Y, self.image_vi_Reg_Cb, self.image_vi_Reg_Cr = RGB2YCrCb(self.image_vi_Reg_RGB)
        ir_stack_reg_Y = torch.cat([self.image_ir_Y, self.image_ir_Y, self.image_ir_Y])
        vi_stack_reg_Y = torch.cat([self.image_vi_Reg_Y, self.image_vi_Y, self.image_vi_warp_Y])
        F_i1, F_i2, F_ib, F_id, F_v1, F_v2, F_vb, F_vd = self.AE_encoder(ir_stack_reg_Y, vi_stack_reg_Y)
        ir_vi_B, ir_vi_D, ir_vi_cross = self.fusion_layer(F_ib, F_vb, F_id, F_vd, F_i2, F_v2)
        fusion_img = self.AE_decoder(ir_vi_B, ir_vi_D, ir_vi_cross)
        self.image_fusion_1, self.image_fusion_gt, self.image_fusion_warp = torch.split(fusion_img, b, dim=0)
        
        # generate_mask
        self.generate_mask()
        self.image_display = torch.cat((self.image_ir_RGB[0:1, 0:1], self.image_ir_warp_RGB[0:1, 0:1]), dim=0).detach()

    def backward_RF_warmup(self):
        # contrastive loss
        self.image_fusion_1_ = self.image_fusion_1.repeat(1, 3, 1, 1) * self.goodmask * self.goodmask_inverse
        self.image_fusion_gt_ = self.image_fusion_gt.repeat(1, 3, 1, 1) * self.goodmask * self.goodmask_inverse
        self.image_fusion_warp_ = self.image_fusion_warp.repeat(1, 3, 1, 1)

        mask_ = torch.logical_and(self.image_ir_Y > 1e-5, self.image_vi_Y > 1e-5)
        mask_ = torch.logical_and(self.image_vi_Reg_Y > 1e-5, mask_)
        mask_ = mask_ * self.goodmask * self.goodmask_inverse

        # photometric loss
        loss_reg_img = self.imgloss(self.image_ir_warp_RGB, self.image_ir_warp_fake_RGB, self.goodmask) + \
                       self.imgloss(self.image_vi_Reg_RGB, self.image_vi_RGB, self.goodmask * self.goodmask_inverse) + \
                       self.imgloss(self.image_fusion_1, self.image_fusion_gt, mask_)

        loss_reg_field = self.weightfiledloss(self.image_vi_warp_RGB, self.image_ir_warp_fake_RGB,
                                              self.deformation_1['ir2vis'], self.disp.permute(0, 3, 1, 2))
        loss_smooth = smoothloss(self.down2) + smoothloss(self.down4) + smoothloss(self.down8)
        loss_border_re = 0.5 * self.border_suppression(self.image_vi_Reg_RGB, self.goodmask_inverse) + \
                         self.border_suppression(self.image_ir_warp_fake_RGB, self.goodmask)

        loss_total = loss_reg_img * 1 + loss_reg_field * 10 + loss_smooth  + loss_border_re * 10
        (loss_total).backward()

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field
        self.loss_smooth = loss_smooth
        self.loss_border = loss_border_re
        self.loss_total = loss_total