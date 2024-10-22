from torchvision import models
from .layers_fusion import *


class AE_Encoder(nn.Module):
    def __init__(self, channel=64):
        super(AE_Encoder, self).__init__()
        self.inception_res_1 = Inception_ResidualBlock(1, channel // 2)
        self.inception_res_2 = Inception_ResidualBlock(channel // 2, channel)
        self.inception_res_3 = Inception_ResidualBlock(channel, channel)

        # common
        self.CATT_ir = ChannelAttention(channel)
        self.SATT_ir = SpatialAttention()
        self.CATT_vi = ChannelAttention(channel)
        self.SATT_vi = SpatialAttention()

        # unique
        self.conv0_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # output
        self.conv_fB_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fB_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)

    def forward(self, ir, vi):
        f0_ir = self.inception_res_1(ir)
        f1_ir = self.inception_res_2(f0_ir)
        f2_ir = self.inception_res_3(f1_ir)

        f0_vi = self.inception_res_1(vi)
        f1_vi = self.inception_res_2(f0_vi)
        f2_vi = self.inception_res_3(f1_vi)

        fB_ir_CATT = f2_ir * self.CATT_ir(f2_ir)
        fB_ir_SATT = f2_ir * self.SATT_ir(f2_ir)
        fB_ir = self.conv_fB_ir(torch.cat([fB_ir_CATT, fB_ir_SATT], 1))

        fB_vi_CATT = f2_vi * self.CATT_vi(f2_vi)
        fB_vi_SATT = f2_vi * self.SATT_vi(f2_vi)
        fB_vi = self.conv_fB_vi(torch.cat([fB_vi_CATT, fB_vi_SATT], 1))

        fD_ir = self.conv0_D1(self.conv1_D1(f2_ir))
        fD_vi = self.conv0_D2(self.conv1_D2(f2_vi))

        return f1_ir, f2_ir, fB_ir, fD_ir, f1_vi, f2_vi, fB_vi, fD_vi


class AE_Decoder(nn.Module):
    def __init__(self, channel=64):
        super(AE_Decoder, self).__init__()
        self.cov5 = nn.Sequential(nn.Conv2d(channel * 3, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cov6 = nn.Sequential(nn.Conv2d(channel * 1, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cov7 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channel * 1, channel // 2, 3, padding=0),
                                  nn.Conv2d(channel // 2, 1, 3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, feature_B, feature_D, feature_cross):
        Output1 = self.cov5(torch.cat([feature_B, feature_D, feature_cross], 1))
        Output2 = self.cov6(Output1)
        Output3 = self.cov7(Output2)
        return Output3


class Fusion_layer(nn.Module):
    def __init__(self, channel=64):
        super(Fusion_layer, self).__init__()
        self.conv0_ir_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)

        self.conv0_ir_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_vi_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_vi_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_vi_D = AFF()

        self.Cross_ir_vi = CrossAttention(channel)

    def forward(self, ir_B, vi_B, ir_D, vi_D, ir_2, vi_2):
        # common
        conv0_ir_B = self.conv0_ir_B(ir_B) + ir_B
        conv0_vi_B = self.conv0_vi_B(vi_B) + vi_B
        conv1_ir_B = self.conv1_ir_B(conv0_ir_B)
        conv1_vi_B = self.conv1_vi_B(conv0_vi_B)
        ir_vi_B = self.conv1_ir_vi_B(conv1_ir_B + conv1_vi_B)

        # unique
        conv0_ir_D = self.conv0_ir_D(ir_D) + ir_D
        conv0_vi_D = self.conv0_vi_D(vi_D) + vi_D
        conv1_ir_D = self.conv1_ir_D(conv0_ir_D)
        conv1_vi_D = self.conv1_vi_D(conv0_vi_D)
        ir_vi_D = self.conv1_ir_vi_D(conv1_ir_D, conv1_vi_D)

        cross_att_ir, cross_att_vi = self.Cross_ir_vi(ir_2, vi_2)
        ir_crossATT = ir_2 * cross_att_ir
        vi_crossATT = vi_2 * cross_att_vi
        ir_vi_cross = ir_crossATT + vi_crossATT

        return ir_vi_B, ir_vi_D, ir_vi_cross


class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.pre_backbone = models.resnet18(pretrained=True)
        self.pre_backbone = nn.Sequential(*list(self.pre_backbone.children())[:-1])
        self.custom_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        features = self.pre_backbone(x)
        features = features.view(features.size(0), -1)
        output = self.custom_fc(features)
        return output


class ResNetSimCLR_2(nn.Module):
    def __init__(self, out_dim):
        super(ResNetSimCLR_2, self).__init__()
        self.pre_backbone = models.resnet18(pretrained=True)
        self.pre_backbone = nn.Sequential(*list(self.pre_backbone.children())[:-1])

    def forward(self, x):
        features = self.pre_backbone(x)
        features = features.view(features.size(0), -1)
        return features


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
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
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGInfoNCE(nn.Module):
    def __init__(self, model_vgg19=Vgg19()):
        super(VGGInfoNCE, self).__init__()
        self.vgg = model_vgg19.cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.args = args
        self.cl_layer = [0,1,2,3]

    def infer(self, x):
        return self.vgg(x)

    def forward(self, sr, hr, lr):
        sr_features = self.vgg(sr)
        if not isinstance(hr, list):
            hr = [hr, ]
        if not isinstance(lr, list):
            lr = [lr, ]
        loss = self.infoNCE(sr_features, hr, lr)
        return loss

    def infoNCE(self, sr_features, hr, lr):
        n_hr_features = []
        n_lr_features = []

        with torch.no_grad():
            for s_hr in hr:
                n_hr_features.append(self.infer(s_hr))

            for s_lr in lr:
                n_lr_features.append(self.infer(s_lr))

        infoNCE_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_layer = sr_features[idx]

            hr_layers = []
            for hr_features in n_hr_features:
                hr_layers.append(hr_features[idx])

            lr_layers = []
            for lr_features in n_lr_features:
                lr_layers.append(lr_features[idx])

            nce_loss = self.l1_nce(sr_layer, hr_layers, lr_layers)
            infoNCE_loss += nce_loss

        return infoNCE_loss / len(self.cl_layer)

    def nce(self, sr_layer, hr_layers, lr_layers,temp=1):
        loss = 0

        neg_logits = []
        for f_lr in lr_layers:
            neg_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_lr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            neg_logits.append(neg_diff/temp)

        for f_hr in hr_layers:
            pos_logits = []
            pos_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_hr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            pos_logits.append(pos_diff/temp)

            cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            loss += cl_loss
        return loss / len(hr_layers)

    def lmcl_loss(self, logits):
        """
        logits: BXK, the first column is the positive similarity
        """
        pos_sim = logits[0]
        pos_logits = pos_sim.exp()  # Bx1

        # neg_sim = torch.cat(logits[1:], dim=1)
        neg_sim = torch.cat(logits, dim=1)

        print('neg:',neg_sim.exp())
        neg_logits = torch.sum(neg_sim.exp(), dim=1, keepdim=True)  # Bx1
        print('neg:', neg_logits)
        print('pos:', pos_logits)
        loss = -torch.log(pos_logits / neg_logits).mean()
        return loss

    def l1_nce(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []
        for f_lr in lr_layers:
            neg_diff = torch.abs(sr_layer - f_lr).mean(dim=[-3, -2, -1]).unsqueeze(1) /0.02  #0.05
            # neg_logits.append(neg_diff.exp())
            # neg_logits.append(1/(neg_diff+1e-8))
            # neg_logits.append(-(neg_diff))
            neg_logits.append(neg_diff)

        for f_hr in hr_layers:
            pos_logits = []
            pos_diff = torch.abs(sr_layer - f_hr).mean(dim=[-3, -2, -1]).unsqueeze(1) /0.02  #0.05
            # pos_logits.append(pos_diff.exp())
            # pos_logits.append(1/(pos_diff+1e-8))
            # pos_logits.append(-(pos_diff)+0.5)
            pos_logits.append(pos_diff)

            # logits = torch.cat(pos_logits + neg_logits, dim=1)
            # cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device,
            #                                                   dtype=torch.long))  # self.ce_loss(logits)
            neg_logits = torch.cat(neg_logits, dim=1).mean(dim=1, keepdim=True)
            cl_loss = torch.mean(pos_logits[0] / neg_logits)

            # cl_loss = self.lmcl_loss(pos_logits + neg_logits)

            # elif self.args.cl_loss_type == 'LMCL':
            #     cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            # else:
            #     raise TypeError(f'{self.args.cl_loss_type} is not found in loss/cl.py')
            loss += cl_loss

        return loss / len(hr_layers)


if __name__ == '__main__':
    ir_B, vi_B, ir_D, vi_D = torch.randn([4, 64, 256, 256]), torch.randn([4, 64, 256, 256]), torch.randn(
        [4, 64, 256, 256]), torch.randn([4, 64, 256, 256])
    Fu = Fusion_layer()
    ir_vi_B, ir_vi_D = Fu(ir_B, vi_B, ir_D, vi_D)
    print(ir_vi_B.shape)
    print(ir_vi_D.shape)
