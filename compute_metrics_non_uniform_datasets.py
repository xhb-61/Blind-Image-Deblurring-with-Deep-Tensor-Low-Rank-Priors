import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy.io import savemat

from metrics.utils.metrics import aver_bmp_psnr_ssim_par, aver_bmp_psnr_ssim_rot_par
import argparse
from glob import glob
import os
from matplotlib.image import imread
import cv2
import numpy as np
from skimage import transform
from metrics.utils.imtools import rgb2ycbcr, alignment
import pickle
from metrics.utils.imtools import rgb2gray
from metrics.utils.imtools import imshow


parser = argparse.ArgumentParser(description='compute PSNR')
parser.add_argument('--parallel', default=False, help='Parallel computation of the PSNR')
parser.add_argument('--dataset_name', default='Lai' ,choices=['Levin','Lai','Kohler'])
parser.add_argument('-C', default=3)
parser.add_argument('-deal_boundary', default=False)
args = parser.parse_args()

if args.dataset_name == 'Lai':
    # sharp_name = ['manmade']
    sharp_name = ['manmade','natural','people','saturated','text']
    num_image = 5
    num_kernels = 4
    sharp_folder = './datasets/lai/ground_truth/'
    # sharp_folder = './real1/'
    recover_folder = './results_2tucker_newname/'
    # recover_folder = './my1/'
else:
    pass

# model_list = ['12_whyte']
model_list = ['my']

# model_list = ['09_cho', '10_xu', '11_krishnan', '11_levin', '13_sun', '13_xu_uniform','13_zhang','13_zhong','14_michaeli', '14_pan', '14_perrone']
# recover = './18Tao_SRN_BD/Nonuniform_Lai/'

# recover_folder = '../19Kupyn_DeblurGANv2/Nonuniform_Lai'
# recover_folder = '../18Tao_SRN_BD/Nonuniform_Lai/'
# recover_folder = '../15Sun_LCNN/Lai_NonUniform/'

for model in model_list:
    print(model)
    for name in sharp_name:
        sp = []
        rc = []
        for i in range(num_image):
            for j in range(num_kernels):
                sp_dir = os.path.join(sharp_folder, '%s_0%d'%(name, i+1))
                rc_dir = os.path.join(recover_folder, '%s_0%d_gyro_0%d_x'%(name, i+1, j+1))
                # rc_dir = os.path.join(recover_folder, '%s_0%d_gyro_0%d' % (name, i + 1, j + 1))
                sp_file = sorted(glob(sp_dir + '*.png'))
                rc_file = sorted(glob(rc_dir + '*.png'))

                # print('sp_file',sp_file)
                # print('rc_file',rc_file)

                sp_img = imread(sp_file[0])
                rc_img = imread(rc_file[0])

                # print('sp_img:', sp_img.shape)
                # print('rc_img:', rc_img.shape)

                if args.deal_boundary:
                    kernel_size = [25, 25]
                    img_size = sp_img.shape[0:2]
                    rc_img = rc_img[kernel_size[0] // 2:kernel_size[0] // 2 + img_size[0], kernel_size[1] // 2:kernel_size[1] // 2 + img_size[1],:]
                sp_img = sp_img[...,0:3]

                h,w,c = sp_img.shape
                rc_img = rc_img[:h,:w,:c]

                if not np.array_equal(sp_img.shape,rc_img.shape):
                    print('%s_0%d_gyro_0%d' % (name, i + 1, j + 1))
                    print(sp_img.shape)
                    print(rc_img.shape)


                sp.append(sp_img)
                rc.append(rc_img)
        print('end reading image')
        # print(len(rc))
        # asss
        output = aver_bmp_psnr_ssim_rot_par(rc, sp, bd_cut=15, maxshift=14, maxangle=0.5, num_cores=0,
                                            shift_inter=1, angle_inter=0.1, ssim_compute=True,
                                            verbose=True, show_aligned=True)
        print(recover_folder)
        print(model, name, output['PSNR_mean'])
        print(model, name, output['SSIM_mean'])
        os.makedirs('./comparison/%s/'%model,exist_ok=True)
        os.makedirs('./comparison/%s/'%model,exist_ok=True)
        psnr_result = np.zeros((num_image, num_kernels))
        ssim_result = np.zeros((num_image, num_kernels))
        for i in range(num_image):
            for j in range(num_kernels):
                idx = i * num_kernels + j
                imshow(output['I1_matched'][idx], dir='./comparison/%s/'%model, str='aligned_%s_0%d_gyro_0%d'%(name, i+1, j+1))
                imshow(output['I2_matched'][idx], dir='./comparison/%s/'%model, str='gt_%s_0%d_gyro_0%d'%(name, i+1, j+1))
                psnr_result[i, j] = output['PSNR'][idx]
                ssim_result[i, j] = output['SSIM'][idx]
        print(model, name, psnr_result)
        savemat('./comparison/%s/%s.mat' % (model, name), {'PSNR': psnr_result, 'SSIM': ssim_result})

        # print(model, name, output['PSNR'])
        # dict = {'I1': output["I1_matched"], 'I2': output["I2_matched"]}
        # savemat(recover + model + "/%s_%s.mat"%(model, name), dict)
        # f = open(recover + model + "/%s_%s.pkl"%(model, name), "wb")
        # pickle.dump(dict, f)
        # f.close()
