import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layers_reg import *
import kornia.utils as KU
import kornia.filters as KF
import os
import torch.nn.init as init

class SpatialTransformer(nn.Module):
    def __init__(self, h, w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h, w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1] == 2:
            disp = disp.permute(0, 2, 3, 1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1], disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False), flow


class DispEstimator(nn.Module):
    def __init__(self, channel, depth=4, norm=nn.BatchNorm2d, dilation=1):
        super(DispEstimator, self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel, channel, 3, act=None, norm=None, dilation=dilation, padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, act=None))
        # self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel + self.corrks ** 2
        dilation = 1
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256, 256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
        # self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        # self.AP3=nn.AvgPool2d(3,stride=1,padding=1)

    def localcorr(self, feat1, feat2):
        feat = self.featcompressor(torch.cat([feat1, feat2], dim=1))
        b, c, h, w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1, (13, 13), (3, 3), border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth, kernel_size=self.corrks, dilation=4, padding=2 * (self.corrks - 1),
                                 stride=1).reshape(b, c, -1, h, w)
        localcorr = (feat2.unsqueeze(2) - feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat, localcorr], dim=1)
        return corr

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0, 1, 0, 0] != w - 1 or self.scale[0, 0, 0, 0] != h - 1:
            self.scale = torch.FloatTensor([w, h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1, feat2)
        for i, layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr, (13, 13), (3, 3), border_type='replicate')
        disp = corr.clamp(min=-300, max=300)
        return disp / self.scale


class DispRefiner(nn.Module):
    def __init__(self, channel, dilation=1, depth=4):
        super(DispRefiner, self).__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None))
        oc = channel
        ic = channel + 2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(
                Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)

    def forward(self, feat1, feat2, disp):
        b = feat1.shape[0]
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))
        corr = torch.cat([feat, disp], dim=1)
        delta_disp = self.estimator(corr)
        disp = disp + delta_disp
        return disp


class Feature_extractor_unshare(nn.Module):
    def __init__(self, depth, base_ic, base_oc, base_dilation, norm):
        super(Feature_extractor_unshare, self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i % 2 == 1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(
                    ResConv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            else:
                feature_extractor.append(
                    Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            ic = oc
            # if i%2==1 and i<depth-1:
            #     oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class registration_net(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2):
        super(registration_net, self).__init__()
        self.num_pyramids = num_pyramids
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth, base_ic=64, base_oc=64,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth, base_ic=64, base_oc=64,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        # self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 1, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 1, base_oc * 1, kernel_size=3, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(base_oc * 1, base_oc * 1, kernel_size=3, stride=1, padding=2, dilation=2, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 1, base_oc * 1, kernel_size=3, stride=2, padding=2, dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(base_oc * 1, base_oc * 1, kernel_size=3, stride=1, padding=4, dilation=4, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 1, base_oc * 1, kernel_size=3, stride=2, padding=4, dilation=4, norm=nn.InstanceNorm2d))
        self.matcher1 = DispEstimator(base_oc * 1, matcher_depth, dilation=4)
        self.matcher2 = DispEstimator(base_oc * 1, matcher_depth, dilation=2)
        self.refiner = DispRefiner(base_oc * 1, 1)
        self.grid_down = KU.create_meshgrid(64, 64).cuda()
        self.grid_full = KU.create_meshgrid(128, 128).cuda()
        self.scale = torch.FloatTensor([128, 128]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1

    def match(self, feat11, feat12, feat21, feat22, feat31, feat32):
        # compute scale (w,h)
        if self.scale[0, 1, 0, 0] * 2 != feat11.shape[2] - 1 or self.scale[0, 0, 0, 0] * 2 != feat11.shape[3] - 1:
            self.h, self.w = feat11.shape[2], feat11.shape[3]
            self.scale = torch.FloatTensor([self.w, self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat11.device)

        # deformation field estimation
        disp2_raw = self.matcher2(feat31, feat32)
        disp2 = F.interpolate(disp2_raw, [feat21.shape[2], feat21.shape[3]], mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2], feat21.shape[3]).cuda()
        # re-sampling correction
        feat21 = F.grid_sample(feat21, self.grid_down + disp2.permute(0, 2, 3, 1))

        disp1_raw = self.matcher1(feat21, feat22)
        disp1 = F.interpolate(disp1_raw, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        disp2 = F.interpolate(disp2, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2], feat11.shape[3]).cuda()
        feat11 = F.grid_sample(feat11, self.grid_full + (disp1 + disp2).permute(0, 2, 3, 1))

        disp_scaleup = (disp1 + disp2) * self.scale
        disp = self.refiner(feat11, feat12, disp_scaleup)
        disp = KF.gaussian_blur2d(disp, (17, 17), (5, 5), border_type='replicate') / self.scale

        if self.training:
            return disp, disp_scaleup / self.scale, disp2
        return disp, disp_scaleup / self.scale, disp2

    def forward(self, src, tgt, type='ir2vis'):
        # Multi-level Feature Extractor
        b, c, h, w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)
        feat0 = torch.cat([feat01, feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11, feat12 = feat1[0:b], feat1[b:]
        feat21, feat22 = feat2[0:b], feat2[b:]
        feat31, feat32 = feat3[0:b], feat3[b:]

        # Multi-level Deformation Field Estimator
        if type == 'bi':
            disp_12, disp_12_down4, disp_12_down8 = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_21, disp_21_down4, disp_21_down8 = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            t = torch.cat([disp_12, disp_21, disp_12_down4, disp_21_down4, disp_12_down8, disp_21_down8])
            t = F.interpolate(t, [h, w], mode='bilinear')
            down2, down4, donw8 = torch.split(t, 2 * b, dim=0)
            disp_12_, disp_21_ = torch.split(down2, b, dim=0)
        elif type == 'ir2vis':
            disp_12, disp_12_down4, disp_12_down8 = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            t = torch.cat([disp_12, disp_12_down4, disp_12_down8])
            t = F.interpolate(t, [h, w], mode='bilinear')
            down2, down4, donw8 = torch.split(t, 1 * b, dim=0)
            disp_12_ = down2
        elif type == 'vis2ir':
            disp_21, disp_21_down4, disp_21_down8 = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            t = torch.cat([disp_21, disp_21_down4, disp_21_down8])
            t = F.interpolate(t, [h, w], mode='bilinear')
            down2, down4, donw8 = torch.split(t, 1 * b, dim=0)
            disp_21_ = down2

        if type == 'bi':
            return {'ir2vis': disp_12_, 'vis2ir': disp_21_, 'down2': down2, 'down4': down4, 'down8': donw8}
        elif type == 'ir2vis':
            return {'ir2vis': disp_12_, 'down2': down2, 'down4': down4, 'down8': donw8}
        elif type == 'vis2ir':
            return {'vis2ir': disp_21_, 'down2': down2, 'down4': down4, 'down8': donw8}



def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                   float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass

if __name__ == '__main__':
    matcher = registration_net().cuda()
    ir = torch.rand(2, 64, 512, 512).cuda()
    vis = torch.rand(2, 64, 512, 512).cuda()
    disp = matcher(ir, vis, 'bi')
