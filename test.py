import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy import misc
import cv2
from utils.dataloader import test_dataset
from lib.model import SOD_pvt
import datetime
import time




parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./Experiments_Results/pvt/Net_epoch_best.pth')
opt = parser.parse_args()

inf_time = 0

for _data_name in['DUT-OMRON', 'DUTS-TE', 'ECSSD','HKU-IS', 'PASCAL-S', 'SOD']:
    start = datetime.datetime.now()
    start = time.time()
    data_path = './Data/SOD/{}'.format(_data_name)
    save_path_pred = './test_result/{}/{}/pred_4/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = SOD_pvt().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path_pred, exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    end = datetime.datetime.now()
    for i in range(test_loader.size):
        ori_image, image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        pred = model(image)[3]

        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred_map = F.sigmoid(pred)
        pred_map = pred_map.data.cpu().numpy().squeeze()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        pred_map = pred_map * 255


        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        # cv2.imwrite(save_path + name,res*255)
        cv2.imwrite(save_path_pred + name, pred_map)
        # cv2.imwrite(save_path_entropy + name, entropy_map)
    
    end = time.time()
    inf_time += (end - start)

inf_per_image = inf_time / 1000
fps = 1 /  inf_per_image
print("inf_per_image: {}, fps: {}".format(inf_per_image, fps))
