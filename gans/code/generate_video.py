# 1. Read the generate videos file
# 2. Print trajectory values of 1-object
# 3. Sample one cell from GAN
# 4. Place it at those trajectory points

from __future__ import print_function
from PIL import Image
import random
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.transforms
import argparse
import os
import random
import numpy as np
from numpy import random

import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import gan, wgan, dcgan
import reconstruction
from custom_dataloaders import ImageFolderWithCache, CompositeImageFolder, read_all_images
from utils import RandomVerticalFlip, weights_init
import glob
import sys

def convert_img(img_tensor, nrow):
    img_tensor = img_tensor.cpu()
    ndarr = img_tensor.mul(0.5).add(0.5).mul(255).byte().numpy()
    ndarr = np.reshape(ndarr,(nrow,1,64,64))
    ndarr = np.transpose(ndarr,(0,2,3,1))
    return ndarr

def place_objects(new_arr,img_arr,mask,traj):

    try:
        cx,cy = traj[0],traj[1]
        w = img_arr.shape[0]
        h = img_arr.shape[1]


        upperleftx = cx - w / 2
        upperlefty = cy - h / 2
        lowerrightx = cx + w / 2
        lowerrighty = cy + h / 2

        '''
        if (upperleftx < 0):
            upperleftx = 0
        elif (upperleftx > 1023):
            upperleftx = 1023
    
        if (upperlefty < 0):
            upperlefty = 0
        elif (upperlefty > 1023):
            upperlefty = 1023
    
        if (lowerrightx < 0):
            lowerrightx = 0
        elif (lowerrightx > 1023):
            lowerrightx = 1023
    
        if (lowerrighty < 0):
            lowerrighty = 0
        elif (lowerrighty > 1023):
            lowerrighty = 1023
        '''
        roi = new_arr[
                int(upperlefty):int(lowerrighty),
                int(upperleftx):int(lowerrightx)
            ]

        #mask[mask == 0] = 145
        mask_inv = np.bitwise_not(mask)
        img1_bg = np.bitwise_and(roi,mask_inv)


        img2_fg = np.bitwise_and(img_arr,mask)

        dst = np.add(img1_bg,img2_fg)
        new_arr[
                int(upperlefty):int(lowerrighty),
                int(upperleftx):int(lowerrightx)
                ] = dst
        #(img_arr OR img_mask)

    except:
        print (sys.exc_info())
        print ('in except')
        pass

    return new_arr

if __name__=='__main__':
    cls = 'normal_100'
    frame_out_path = '/media/narita/Data/Neutrophils/fps_20_frames/average_motion_model/'+cls+'/'

    '''
    original_trajectories_path = '/media/narita/Data/Neutrophils/fps_20_frames/5_07072016_Phagocytosis_DMSO/' \
                                 'final/tracking/organized_kalman_output_center_coords.npy'
    original_trajectories = np.load(original_trajectories_path)
    ofirst = original_trajectories[0,:,:]
    '''
    generated_trajectories_path = '/media/narita/Data/Neutrophils/fps_20_frames/average_motion_model/' \
                                  'ar_generated_center_coords_'+cls+'.npy'

    generated_trajectories = np.load(generated_trajectories_path)
    print (generated_trajectories.shape)

    nframes = generated_trajectories.shape[1]
    nobjects = generated_trajectories.shape[0]

    '''
    # Load generator
    model_path = '/media/narita/Data/neutrophils-gan/models/size-64-64_gan-adam/saved_models/netG_iter_5000.pth'
    netG = dcgan.DCGAN_G((64, 64), 100, 1, 64, 0)
    netG.load_state_dict(torch.load(model_path))
    #print(netG)
    netG.eval()

    noise_batch = torch.FloatTensor(generated_trajectories.shape[0], 100, 1, 1).normal_(0, 1)
    noise_batch = Variable(noise_batch, volatile=True)
    fake_batch = netG(noise_batch)
    all_images = convert_img(fake_batch.data, generated_trajectories.shape[0])
    all_images = all_images[:,:,:,0]
    print (all_images.shape)
    print (generated_trajectories.shape)
    '''

    generated_images_path = '/media/narita/Data/neutrophils-gan/selected-individual-cels'

    generated_images = []

    for f in glob.glob(os.path.join(generated_images_path,'*.png')):
        if not f.endswith('_mask.png'):
            generated_images.append(f)

    get_img_num = lambda img_file: int(os.path.basename(img_file).split('.')[0])
    generated_images = sorted(generated_images,key=get_img_num)
    all_images = []
    all_masks = []

    mylist = (range(len(generated_images)))
    selected_indices = random.choice(mylist,nobjects)
    print (selected_indices)

    for i in range(len(selected_indices)):
        curr_img = generated_images[selected_indices[i]].rsplit('/',1)[1].split('.png')[0]
        mask_path = os.path.join(generated_images_path,curr_img+'_mask.png')
        all_images.insert(i,np.array(Image.open(generated_images[i])))
        all_masks.insert(i,np.array(Image.open(mask_path)))


    new_frames = np.full((nframes,1024,1024),0)
    for i in range(nframes):
        for j in range(nobjects):
            print (j)

            new_frames[i]=place_objects(
                                            new_frames[i],
                                            all_images[j],
                                            all_masks[j],
                                            generated_trajectories[j,i]
                                        )
        
        
        #new_frames[i] = (new_frames[i]*255).astype(np.uint8)
        #im = Image.fromarray(new_frames[i],mode='L')

        #im.save(os.path.join(frame_out_path,str(i)+'.png'))
        #print(i)
        new_img = Image.fromarray((new_frames[i]*255).astype('uint8'))
        new_img.save(os.path.join(frame_out_path,str(i)+'.png'))
        #np.save(os.path.join(frame_out_path,'_'+str(i)+'.npy',new_frames))
        plt.imshow (new_frames[i],cmap='gray')
        plt.title(str(i))
        #plt.show()
        plt.pause(0.01)

