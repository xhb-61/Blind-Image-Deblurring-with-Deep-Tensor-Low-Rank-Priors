''' Metric Computation including PSNR and SSIM '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from scipy.interpolate import interp2d
from skimage.transform import warp, EuclideanTransform
from itertools import product
import time
from metrics.utils.imtools import rgb2gray, torch2np
from joblib import Parallel, delayed
import multiprocessing
from metrics.utils.imtools import imshow

def psnr(img1,img2):
    PIXEL_MAX = 1
    img1 = torch.clamp(img1,min = 0,max = 1)
    img2 = torch.clamp(img2,min = 0,max = 1)
    mse = torch.mean((img1.cpu() - img2.cpu()) ** 2).numpy()
    return 10 * np.log10(PIXEL_MAX **2 / mse)

def aver_psnr(img1,img2):
    ''' For images with same size and stored by a matrix'''
    PSNR = 0
    assert img1.size() == img2.size()
    for i in range(img1.size()[0]):
        PSNR += psnr(img1[i,...], img2[i,...])
    return PSNR / img1.size()[0]

def aver_psnr_ds(img1, img2):
    ''' For images with different size and stored by a list'''
    PSNR = 0
    for i in range(len(img1)):
        PSNR += psnr(img1[i], img2[i])
    return PSNR / len(img1)




def comp_upto_shift(I1 , I2, cut = 15):
    '''The same shift scheme by Levin'''
    I1[I1 < 0] = 0
    I1[I1 > 1] = 1
    I2[I2 > 1] = 1
    I2[I2 < 0] = 0
    maxshift = 5
    shifts = np.arange(-5 ,5.25 ,0.25)
    ssdem = np.zeros((len(shifts),len(shifts)))
    I2= I2[cut:-cut ,cut:-cut]
    I1= I1[cut-maxshift:-cut+maxshift ,cut- maxshift:-cut+maxshift]
    [N1, N2]=np.shape(I2)
    gx = np.arange(1-maxshift , N2+maxshift +1)
    gy = np.arange(1-maxshift , N1+maxshift +1)
    gx0 = np.arange(N2)+1
    gy0 = np.arange(N1)+1

    f = interp2d( gx , gy , I1 )

    for i in range ( 0 ,len(shifts)) :
        for j in range ( 0 ,len(shifts)) :
            gxn = gx0 + shifts[i]
            gyn = gy0 + shifts[j]

            tI1 = f(gxn , gyn)
            ssdem[i,j]=np .sum((tI1-I2) ** 2)
    ssde=min(ssdem.flatten())
    psnr = 20*np.log10(1/np.sqrt(ssde/(N1*N2)))

    return psnr

def bmp_psnr(img1, img2, cut=15, to_int = False):
    '''Using Best matching pixels to compute PSNR for the whole image'''
    img1 = torch.squeeze(img1).numpy()
    img2 = torch.squeeze(img2).numpy()
    if to_int:
        img1 = np.around(img1*255).astype(int) / 255
        img2 = np.around(img2*255).astype(int) / 255
    PSNR = comp_upto_shift(img1, img2, cut=cut)
    return PSNR

def aver_bmp_psnr(img1, img2, cut = 15, to_int = False):
    ''' For images with different size and stored by a list'''
    PSNR = 0
    for i in range(len(img1)):
        PSNR += bmp_psnr(img1[i], img2[i], cut=cut, to_int = to_int)
    return PSNR / len(img1)



### Parallel Computing of PSNR by Best Matching

def comp_upto_shif_algn(I1 , I2, cut = 15, maxshift = 5, ssim_compute = False):
    '''
    Compute the PSNR and SSIM for grayscale image aligned by best matching principle using same shift scheme by Levin \etal's code
        I1: Deblurred results
        I2: Sharp image
        cut: The boundary cut
    '''
    shifts = np.arange(-maxshift ,maxshift+0.25 ,0.25)
    ssdem = np.zeros((len(shifts),len(shifts)))
    I2 = I2[cut:-cut ,cut:-cut]
    I1 = I1[cut-maxshift:-cut+maxshift ,cut- maxshift:-cut+maxshift]
    [N1, N2] = np.shape(I2)
    gx = np.arange(1-maxshift , N2+maxshift +1)
    gy = np.arange(1-maxshift , N1+maxshift +1)
    gx0 = np.arange(N2)+1
    gy0 = np.arange(N1)+1
    f = interp2d(gx , gy , I1)
    for i in range(0 ,len(shifts)):
        for j in range(0 ,len(shifts)):
            gxn = gx0 + shifts[i]
            gyn = gy0 + shifts[j]
            tI1 = f(gxn , gyn)
            ssdem[i,j]=np.sum((tI1-I2) ** 2)
    ssde = min(ssdem.flatten())
    psnr_metric = 20*np.log10(1/np.sqrt(ssde/(N1*N2)))

    ssim_metric = None
    i, j = np.nonzero(ssdem == ssde)
    gxn = gx0+shifts[np.min(i)] # np.min avoids multiple matches
    gyn = gy0+shifts[np.min(j)]
    f = interp2d(gx , gy , I1)

    I1_matched = f(gxn, gyn)
    if ssim_compute:
        ssim_metric = ssim(I1_matched, I2)
    return psnr_metric, ssim_metric, I1_matched, I2.copy()

def comp_upto_shif_algn_color(I1 , I2, cut = 15, maxshift = 5, ssim_compute = False):
    '''
    【Compute the PSNR and SSIM for color image aligned by best matching principle using same shift scheme by Levin \etal's code
        I1: Deblurred results
        I2: Sharp image
        cut: The boundary cut
    '''
    shifts = np.arange(-maxshift ,maxshift + 0.25 ,0.25)
    ssdem = np.zeros((len(shifts),len(shifts)))

    # The best matching is done in grayscale image for color image.
    I1_gray = rgb2gray(I1)
    I2_gray = rgb2gray(I2)
    I2 = I2[cut:-cut ,cut:-cut,:]
    I1 = I1[cut-maxshift:-cut+maxshift ,cut- maxshift:-cut+maxshift,:]

    N1, N2, C = np.shape(I2)
    I2_gray = I2_gray[cut:-cut ,cut:-cut]
    I1_gray = I1_gray[cut-maxshift:-cut+maxshift ,cut- maxshift:-cut+maxshift]


    gx = np.arange(1-maxshift , N2+maxshift +1)
    gy = np.arange(1-maxshift , N1+maxshift +1)
    gx0 = np.arange(N2)+1
    gy0 = np.arange(N1)+1
    f = interp2d(gx , gy , I1_gray)
    for i in range(0 ,len(shifts)) :
        for j in range(0 ,len(shifts)) :
            gxn = gx0 + shifts[i]
            gyn = gy0 + shifts[j]
            tI1 = f(gxn , gyn)
            ssdem[i,j] = np.sum((tI1-I2_gray) ** 2)
    ssde = min(ssdem.flatten())

    # Use the best matching in color mode:
    i, j = np.nonzero(ssdem == ssde)
    gxn = gx0 + shifts[i]
    gyn = gy0 + shifts[j]

    I1_matched = np.zeros_like(I2)
    for chn in range(C):
        f = interp2d(gx , gy , I1[:,:,chn])
        I1_matched[:,:,chn] = f(gxn , gyn)

    ssde = np.sum((I1_matched-I2)**2)
    psnr_metric = 20*np.log10(1/np.sqrt(ssde/(N1*N2*C)))

    ssim_metric = None

    # ssim is also computed in grayscale.
    if ssim_compute:
        ssim_metric = ssim(rgb2gray(I1_matched), I2_gray)
    return psnr_metric, ssim_metric, I1_matched, I2.copy()

def aver_bmp_psnr_ssim_par(img1, img2, num_cores = None, bd_cut = 15, maxshift = 5, ssim_compute = True, show_aligned = False, verbose = False):
    ''' Parallel computing is applied for images'''
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    im_len = len(img1)
    for i in range(im_len):
        img1[i] = np.squeeze(torch2np(img1[i]))
        img2[i] = np.squeeze(torch2np(img2[i]))

        img1[i][img1[i]<0] = 0
        img2[i][img2[i]<0] = 0
        img1[i][img1[i]>1] = 1
        img2[i][img2[i]>1] = 1

        img1[i] = np.around(img1[i]*255).astype(int) / 255
        img2[i] = np.around(img2[i]*255).astype(int) / 255

    if num_cores == 0:
        if len(img1[0].shape) == 3:
            Results = comp_upto_shif_algn_color(img1[0], img2[0], cut=bd_cut, maxshift=maxshift,
                                          ssim_compute=ssim_compute)
        else:
            Results = comp_upto_shif_algn(img1[0], img2[0], cut=bd_cut, maxshift=maxshift,
                                          ssim_compute=ssim_compute)
    else:
        if len(img1[0].shape) == 3:
            # print(num_cores)
            Results  = Parallel(n_jobs=num_cores)(delayed(comp_upto_shif_algn_color)(img1[i], img2[i],
                     cut = bd_cut, maxshift = maxshift, ssim_compute= ssim_compute) for i in range(im_len))
        else:
            Results  = Parallel(n_jobs=num_cores)(delayed(comp_upto_shif_algn)(img1[i], img2[i],
                 cut = bd_cut, maxshift = maxshift, ssim_compute= ssim_compute) for i in range(im_len))

    PSNR = np.zeros((im_len,1))
    for ii in range(im_len):
        try: PSNR[ii] = Results[ii][0]
        except: PSNR[ii] = Results[ii]

    output = {}
    PSNR_mean = np.mean(PSNR)
    output['PSNR_mean'] = PSNR_mean
    if ssim_compute:
        SSIM = np.zeros((im_len, 1))
        for ii in range(im_len):
            SSIM[ii] = Results[ii][1]
        SSIM_mean = np.mean(SSIM)
        output['SSIM_mean'] = SSIM_mean
    if show_aligned:
        I1_matched = [None] * im_len
        I2_matched = [None] * im_len
        for ii in range(im_len):
            I1_matched[ii] = Results[ii][2]
            I2_matched[ii] = Results[ii][3]
        output['I1_matched'] = I1_matched
        output['I2_matched'] = I2_matched
    if verbose:
        output['PSNR'] = PSNR
        if ssim_compute:
            output['SSIM'] = SSIM
    return output


### Parallel Computing of PSNR by Best Matching with Rotation
def aver_bmp_psnr_ssim_rot_par(img1, img2, num_cores = None, bd_cut = 15, maxshift = 10, shift_inter=1, angle_inter=0.1,
                               maxangle=0.5, ssim_compute = True, show_aligned = False, verbose = False):
    ''' Parallel computing is applied for images'''
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    im_len = len(img1)
    for i in range(im_len):
        img1[i] = np.squeeze(torch2np(img1[i]))
        img2[i] = np.squeeze(torch2np(img2[i]))

        img1[i][img1[i]<0] = 0
        img2[i][img2[i]<0] = 0
        img1[i][img1[i]>1] = 1
        img2[i][img2[i]>1] = 1

        img1[i] = np.around(img1[i]*255).astype(int) / 255
        img2[i] = np.around(img2[i]*255).astype(int) / 255

    if num_cores == 0:
        Results = [comp_upto_shif_rot_algn_color(img1[i], img2[i], cut=bd_cut, maxshift=maxshift, maxangle=maxangle,
                                          ssim_compute=ssim_compute, shift_inter=shift_inter, angle_inter=angle_inter) for i in range(im_len)]

    else:
        Results  = Parallel(n_jobs=num_cores)(delayed(comp_upto_shif_rot_algn_color)(img1[i], img2[i],
                     cut = bd_cut, maxshift = maxshift, ssim_compute= ssim_compute, shift_inter=shift_inter, angle_inter=angle_inter) for i in range(im_len))

    PSNR = np.zeros((im_len,1))
    for ii in range(im_len):
        try: PSNR[ii] = Results[ii][0]
        except: PSNR[ii] = Results[ii]
    # print('psnr:',PSNR)
    # print('Results',Results)
    output = {}
    PSNR_mean = np.mean(PSNR)
    output['PSNR_mean'] = PSNR_mean
    if ssim_compute:
        SSIM = np.zeros((im_len, 1))
        for ii in range(im_len):
            SSIM[ii] = Results[ii][1]
        SSIM_mean = np.mean(SSIM)
        output['SSIM_mean'] = SSIM_mean
    if show_aligned:
        I1_matched = [None] * im_len
        I2_matched = [None] * im_len
        for ii in range(im_len):
            I1_matched[ii] = Results[ii][2]
            I2_matched[ii] = Results[ii][3]
        output['I1_matched'] = I1_matched
        output['I2_matched'] = I2_matched
    if verbose:
        output['PSNR'] = PSNR
        if ssim_compute:
            output['SSIM'] = SSIM
    return output

def comp_upto_shif_rot_algn_color(I1 , I2, cut = 15, maxshift = 10, maxangle = 0.5 , shift_inter=1, angle_inter=0.1 ,ssim_compute = False):
    '''
    Compute the PSNR and SSIM for color image aligned by best matching principle using Euclidean transformation
        I1: Deblurred results
        I2: Sharp image
        cut: The boundary cut
    '''
    # Use Euclidean alignment to warp input images
    def psnr(I1, I2):
        N1, N2, C = I1.shape
        ssde = np.sum((I1 - I2) ** 2)
        return 20 * np.log10(1 / np.sqrt( ssde/ (N1 * N2 * C)))

    I1_gray = rgb2gray(I1)
    I2_gray = rgb2gray(I2)
    # Search the best matching principle with relatively small search scope
    x_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    y_shift = np.arange(-maxshift, maxshift + shift_inter, shift_inter)
    r_shift = np.arange(-maxangle, maxangle + angle_inter, angle_inter)
    r_shift = r_shift /180 * np.pi

    I2_gray_cut = I2_gray[cut:-cut ,cut:-cut]
    ssdes = []
    trans = []
    for (x,y,r) in product(x_shift,y_shift,r_shift):
        model = EuclideanTransform(translation=[x,y], rotation=r)
        I1_gray_warped = warp(I1_gray, model.inverse)
        I1_gray_warped = I1_gray_warped[cut:-cut ,cut:-cut]
        ssdes.append(np.sum((I1_gray_warped-I2_gray_cut)**2))
        trans.append((x,y,r))

    idx = np.argmin(ssdes)
    bx, by, br = trans[idx]
    model = EuclideanTransform(translation=[bx, by], rotation=br)
    I1_warped = warp(I1, model.inverse)
    I1_warped_cut = I1_warped[cut:-cut ,cut:-cut,:]
    psnr_metric = psnr(I1_warped_cut, I2[cut:-cut ,cut:-cut,:])
    ssim_metric = None

    # ssim is also computed in grayscale.
    if ssim_compute:
        ssim_metric = ssim(rgb2gray(I1_warped_cut), I2_gray_cut)
    return psnr_metric, ssim_metric, I1_warped_cut, I2[cut:-cut ,cut:-cut,:].copy()


import numpy
from scipy import signal

def ssim(img1, img2, cs_map=False):
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
    if np.max(img1) < 2:
        img1 = img1 * 255
        img2 = img2 * 255

    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x, y = numpy.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = numpy.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()
