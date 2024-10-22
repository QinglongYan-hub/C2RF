import os
from dataset import Fusion_Data
from options import TrainOptions
import torchvision
import torch.optim as optim
import scipy.io as scio
from modules.FusionNet import AE_Encoder, AE_Decoder,Fusion_layer
from modules.losses import *
from utils.utils import RGB2YCrCb, YCbCr2RGB
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()

    # 1.modal
    is_cuda = torch.cuda.is_available()
    lr = 1e-4
    epochs = opts.n_ep_Fu
    epochs_gap = 50
    AE_Encoder = AE_Encoder().train()
    AE_Decoder = AE_Decoder().train()
    fusion_net = Fusion_layer().eval()
    if is_cuda == True:
        AE_Encoder.cuda()
        AE_Decoder.cuda()
        fusion_net = fusion_net.cuda()
    optimizer1 = optim.Adam(AE_Encoder.parameters(), lr=lr)
    optimizer2 = optim.Adam(AE_Decoder.parameters(), lr=lr)
    optimizer_fu = optim.Adam(fusion_net.parameters(), lr=lr)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs_gap // 3, epochs_gap // 3 * 2], gamma=0.2)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [epochs_gap // 3, epochs_gap // 3 * 2], gamma=0.2)
    scheduler_fu = torch.optim.lr_scheduler.MultiStepLR(optimizer_fu, [100 // 3+epochs_gap, 100 // 3 * 2+epochs_gap], gamma=0.9)

    # 2.data
    dataset = Fusion_Data(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=opts.nThreads)
    # 3.loss
    MSELoss = nn.MSELoss()
    SmoothL1Loss = nn.SmoothL1Loss()
    L1Loss = nn.L1Loss()
    ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    criteria_fusion = Fusionloss()

    # 4. Training
    save_dir = './result/train/' + opts.dataset
    print(save_dir)
    for save_name in ['fu', 'weight','decom','recon']:
        os.makedirs(os.path.join(save_dir, save_name), exist_ok=True)

    print('============ Training Begins ===============')
    for iteration in range(epochs):
        for step, (ir, vi, vi_Cb, vi_Cr, img_name) in enumerate(train_loader):
            if len(ir.shape) > 4:
                ir, vi, vi_Cb, vi_Cr = ir.squeeze(1), vi.squeeze(1), vi_Cb.squeeze(1), vi_Cr.squeeze(1)
            if is_cuda:
                ir, vi, vi_Cb, vi_Cr = ir.cuda(), vi.cuda(), vi_Cb.cuda(), vi_Cr.cuda()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if iteration < epochs_gap: #Phase I
                feature_I_1, feature_I_2, feature_I_B, feature_I_D, feature_V_1, feature_V_2, feature_V_B, feature_V_D = AE_Encoder(
                    ir, vi)
                img_recon_vi = AE_Decoder(feature_V_B, feature_V_D, feature_V_2)
                img_recon_ir = AE_Decoder(feature_I_B, feature_I_D, feature_I_2)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                mse_loss_B = L1Loss(feature_I_B, feature_V_B)
                loss_decomp = 1 * (cc_loss_D - cc_loss_B + torch.tanh(mse_loss_B))

                mse_loss_VF = 5 * ssim(vi, img_recon_vi) + MSELoss(vi, img_recon_vi)
                mse_loss_IF = 5 * ssim(ir, img_recon_ir) + MSELoss(ir, img_recon_ir)
                Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(vi),
                                       kornia.filters.SpatialGradient()(img_recon_vi))
                Gradient_loss_ir = L1Loss(kornia.filters.SpatialGradient()(ir),
                                          kornia.filters.SpatialGradient()(img_recon_ir))

                loss = 5 * mse_loss_VF + 5 * mse_loss_IF + 10 * Gradient_loss + 10 * Gradient_loss_ir  + 0.01 * loss_decomp
                loss.backward()
                clip_grad_norm_value = 0.01
                nn.utils.clip_grad_norm_(
                    AE_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    AE_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()

                print('Epoch/step: %d/%d, loss: %.5f, lr: %5f' % (
                    iteration + 1, step + 1, loss.item(), optimizer1.state_dict()['param_groups'][0]['lr']))
                print(mse_loss_VF.item(), mse_loss_IF.item(), Gradient_loss.item(), loss_decomp.item())

                if (iteration + 1) % 10 == 0:
                    torch.save(AE_Encoder.state_dict(),
                               os.path.join(save_dir, 'weight/', 'Encoder_ep%d.pth' % (iteration + 1)))
                    torch.save(AE_Decoder.state_dict(),
                               os.path.join(save_dir, 'weight/', 'Decoder_ep%d.pth' % (iteration + 1)))
            
            else:  # Phase II
                AE_Encoder = AE_Encoder.eval()
                AE_Decoder = AE_Decoder.eval()
                fusion_net = fusion_net.train()
                optimizer_fu.zero_grad()

                feature_I_1, feature_I_2, feature_I_B, feature_I_D, feature_V_1, feature_V_2, feature_V_B, feature_V_D = AE_Encoder(ir, vi)
                ir_vi_B, ir_vi_D, ir_vi_cross = fusion_net(feature_I_B, feature_V_B, feature_I_D, feature_V_D,feature_I_2, feature_V_2)
                img_fusion = AE_Decoder(ir_vi_B, ir_vi_D, ir_vi_cross)

                loss_fusion, loss_in, loss_grad, loss_ssim = criteria_fusion(vi, ir, img_fusion)
                loss = loss_fusion

                print('Epoch/step: %d/%d, loss: %.5f, lr: %5f' % (
                    iteration + 1, step + 1, loss.item(), optimizer_fu.state_dict()['param_groups'][0]['lr']))
                print(loss_ssim.item(), loss_in.item(), loss_grad.item())

                loss.backward()
                optimizer_fu.step()

                if (iteration + 1) % 10 == 0:
                    ir_RGB = torch.cat([ir, ir, ir], dim=1).cuda()
                    vi_RGB = YCbCr2RGB(vi.cuda(), vi_Cb, vi_Cr)
                    img_fusion_RGB = YCbCr2RGB(img_fusion.cuda(), vi_Cb, vi_Cr)
                    for i in range(vi.shape[0]):
                        row1 = torch.cat((ir_RGB[i:i + 1, ::], vi_RGB[i:i + 1, ::], img_fusion_RGB[i:i + 1, ::]), 3)
                        img_filename = os.path.join(save_dir, 'fu/') + 'ep%04d_%s' % (iteration + 1, img_name[i])
                        torchvision.utils.save_image(row1, img_filename, nrow=1)
                if (iteration + 1) % 10 == 0:
                    torch.save(fusion_net.state_dict(),
                               os.path.join(save_dir, 'weight/', 'Fusion_layer_ep%d.pth' % (iteration + 1)))
        if iteration < epochs_gap:  # Phase I
            scheduler1.step()
            scheduler2.step()
        else: # Phase II
            scheduler_fu.step()