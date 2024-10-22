import torch
import os
from model import C2RF
from dataset import TestData, Generate_Reg_Data, imsave
from tqdm import tqdm
from options import TrainOptions
from scipy import io
from utils.utils import checkboard
import torchvision.transforms.functional as TF
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()

    # =================================================================================================
    # Generate new test data
    # =================================================================================================
    # print('============ Generating Begins ===============')
    # gene_dataset = Generate_Reg_Data(opts)
    # gene_dataloader = torch.utils.data.DataLoader(
    #     gene_dataset, batch_size=1, shuffle=False, num_workers=opts.nThreads)
    # p_bar = tqdm(enumerate(gene_dataloader), total=len(gene_dataloader))
    # for idx, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in p_bar:
    #     pass

    # 1.modal
    model = C2RF().cuda()
    model_path = './checkpoint/%s/' % (opts.dataset)
    model.AE_encoder.load_state_dict(torch.load(os.path.join(model_path, 'Encoder.pth')))
    model.AE_decoder.load_state_dict(torch.load(os.path.join(model_path, 'Decoder.pth')))
    model.fusion_layer.load_state_dict(torch.load(os.path.join(model_path, 'Fusion_layer.pth')))
    model.reg_net.load_state_dict(torch.load(os.path.join(model_path, 'RegNet.pth')))
    model.eval()

    # data
    test_dataloader = TestData(opts)
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    # test
    print('============ Testing Begins ===============')
    save_dir = './result/test/%s/' % (opts.dataset)
    for save_name in ['reg_cb', 'reg_vi', 'reg_fu', 'reg_flow']:
        os.makedirs(os.path.join(save_dir, save_name), exist_ok=True)
    for idx, [ir, vi, vi_warp, name] in p_bar:
        vi, ir, vi_warp = vi.cuda(), ir.cuda(), vi_warp.cuda()
        with torch.no_grad():
            b = vi_warp.shape[0]
            vi_reg, fusion_reg, disp, flow = model.forward(vi_warp, ir, vi)

            # save checkboard
            checkboard_image = checkboard(ir, vi_reg)
            checkboard_save_name = os.path.join(save_dir, 'reg_cb',name)
            TF.to_pil_image(checkboard_image.squeeze(0)).save(checkboard_save_name)

            # save reg_vi
            reg_vi_save_name = os.path.join(save_dir, 'reg_vi',name)
            imsave(vi_reg, reg_vi_save_name)

            # save reg_fu
            reg_fu_save_name = os.path.join(save_dir, 'reg_fu',name)
            imsave(fusion_reg, reg_fu_save_name)
            
            # save reg_flow
            reg_flow_save_name = os.path.join(save_dir, 'reg_flow',name)
            io.savemat(reg_flow_save_name.replace('png', 'mat'), {'flow': flow.cpu().numpy()})
    
    

