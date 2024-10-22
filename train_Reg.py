import os
import torch
from dataset import *
from options import TrainOptions
from model import C2RF
from utils.saver import Saver
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main_RF(opts):
    # --- Model Initialization ---
    model = C2RF(opts)
    model.setgpu(opts.gpu)
    model_path = './checkpoint/%s/' % (opts.dataset)
    model.fusion_fixed(model_path)
    print('\n--- Model loaded successfully ---')

    # --- Data Loading ---
    dataset = Registration_Data(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    print('\n--- Data loaded successfully ---')
    
    # --- Training Initialization ---
    ep0 = -1
    total_it = 0
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # --- Display and Output Saver ---
    saver = Saver(opts)

    # --- Training ---
    for ep in range(ep0, opts.n_ep):
        dataset.set_epoch(ep)
        model.setg_epoch(ep)

        if ep <= 1500:
            for it, (image_vi, image_ir, image_vi_warp, image_ir_warp, deformation) in enumerate(train_loader):
                tensors = [image_ir, image_vi, image_ir_warp, image_vi_warp, deformation]
                tensors = [tensor.cuda().detach() for tensor in tensors]
                tensors = [tensor.squeeze(1) if len(tensor.shape) > 4 else tensor for tensor in tensors]
                (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) = tensors
                # update model
                model.update_RF_warmup(image_ir, image_vi, image_ir_warp,image_vi_warp, deformation)

                if not opts.no_display_img:
                    saver.write_display(total_it, model)
                if (total_it + 1) % 10 == 0:
                    Reg_Img_loss = model.loss_reg_img
                    Reg_Field_loss = model.loss_reg_field
                    Total_loss = model.loss_total
                    print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                        total_it, ep, it, model.reg_net_opt.param_groups[0]['lr'], Total_loss))
                    print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}'.format(Reg_Img_loss, Reg_Field_loss))
                total_it += 1

        else:
            for it, (image_vi,image_ir,image_vi_warp,image_ir_warp,
                    image_vi_warp_2,image_ir_warp_2,image_vi_warp_3,image_ir_warp_3,
                    deformation,deformation_2,deformation_3) in enumerate(train_loader):
                tensors = [image_ir, image_vi, image_ir_warp, image_vi_warp, deformation,
                        image_ir_warp_2, image_vi_warp_2, deformation_2,
                        image_ir_warp_3, image_vi_warp_3, deformation_3]
                tensors = [tensor.cuda().detach() for tensor in tensors]
                tensors = [tensor.squeeze(1) if len(tensor.shape) > 4 else tensor for tensor in tensors]
                (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation,
                image_ir_warp_2, image_vi_warp_2, deformation_2,
                image_ir_warp_3, image_vi_warp_3, deformation_3) = tensors

                # update model
                model.update_RF(image_ir, image_vi, image_ir_warp,image_vi_warp, deformation,
                                image_ir_warp_2,image_vi_warp_2, deformation_2,
                                image_ir_warp_3,image_vi_warp_3, deformation_3)
                
                if not opts.no_display_img:
                    saver.write_display(total_it, model)

                if (total_it + 1) % 10 == 0:
                    Reg_Img_loss = model.loss_reg_img
                    Reg_Field_loss = model.loss_reg_field
                    Contra_loss = model.loss_contra
                    Total_loss = model.loss_total
                    print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                        total_it, ep, it, model.reg_net_opt.param_groups[0]['lr'], Total_loss))
                    print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, Contra_loss: {:.4}'.format(
                        Reg_Img_loss, Reg_Field_loss, Contra_loss))
                total_it += 1

        if ((ep + 1) % opts.img_save_freq == 0) or ((ep + 1) == opts.n_ep):
            saver.display_img(ep + 1, it, model, opts, dataset.ir_list)
        if opts.n_ep_decay > -1:
            model.update_lr()
        if ((ep + 1) % opts.model_save_freq == 0) or ((ep + 1) == opts.n_ep):
            saver.write_model(ep + 1, opts.n_ep, model)
    return


if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()
    main_RF(opts)