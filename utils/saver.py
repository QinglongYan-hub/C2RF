import os
import torch
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import torch.nn.functional as F
# tensor to PIL Image
def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def tensor2content(content):
    img = content[0].cpu().float().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))

def save_img_single(img, name):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(name)

def save_content(content, dir):
    for i in range(content.shape[1]):
        sub_content = tensor2content(content[:, i, :, :])
        img = Image.fromarray(sub_content)
        img.save(os.path.join(dir, '%03d.jpg' % (i + 1)))

class Saver():
    def __init__(self, opts):
        # self.display_dir = os.path.join(opts.display_dir, opts.dataset)
        # save_dir = './result/train/' + opts.dataset

        self.display_dir = os.path.join(opts.display_dir, opts.dataset)
        save_dir = os.path.join('./result/train/', opts.dataset)

        for save_name in [opts.img_dir, opts.model_dir]:
            os.makedirs(os.path.join(save_dir, save_name), exist_ok=True)
        self.model_dir = os.path.join(save_dir, opts.model_dir)
        self.image_dir = os.path.join(save_dir, opts.img_dir)
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(
                getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)
            # write img
            image_dis = torchvision.utils.make_grid(
                model.image_display, nrow=model.image_display.size(0)//2)
            self.writer.add_image('Image', image_dis, total_it)


    def display_img(self, ep, it, model, opts, list):
        images_ir, images_vi, images_ir_warp, images_vi_warp, images_vi_Reg, \
        images_fusion_1, images_fusion_gt, images_fusion_warp,goodmask, goodmask_inverse = model.outputs_image()

        for i in range(images_ir.shape[0]):
            row1 = torch.cat(
                (images_ir[i:i + 1, ::],
                 images_vi_warp[i:i + 1, ::],
                 images_fusion_warp[i:i + 1, ::],
                 goodmask[i:i + 1, ::]
                 ), 3)
            row2 = torch.cat(
                (images_vi[i:i + 1, ::],
                 images_vi_Reg[i:i + 1, ::],
                 images_fusion_1[i:i + 1, ::],
                 goodmask_inverse[i:i + 1, ::]
                 ), 3)
            row12 = torch.cat((row1, row2), 2)
            id = it * opts.batch_size + i + 1
            img_filename = '%s/ep%04d_id%04d.jpg' % (self.image_dir, ep, id)
            torchvision.utils.save_image(row12, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_ep, model, best=False):
        mode_save_path = '%s/%s_ep%d.pth' % (self.model_dir, 'Best_RegFusion', ep)
        if best:
            print('--- save the [best]model @ ep %d ---' % (ep))
            mode_save_path = '%s/%s.pth' % (self.model_dir, 'Best_RegFusion')
            model.save(mode_save_path, ep, total_ep)
        else:
            if ep % self.model_save_freq == 0:
                print('--- save the [freq]model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s_ep%d.pth' % (self.model_dir, 'RegNet', ep)
                model.save(mode_save_path, ep, total_ep)

            elif ep == total_ep or ep == total_ep - 1:
                print('--- save the [last]model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s_ep%d.pth' % (self.model_dir, 'RegNet', ep)
                model.save(mode_save_path, ep, total_ep)
            elif ep == -1:
                print('--- save the [first]model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s_ep%d.pth' % (self.model_dir, 'RegNet', ep)
                model.save(mode_save_path, ep, total_ep)
        return