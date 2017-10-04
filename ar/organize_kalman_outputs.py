'''
1. Read the trajectories outputted by Kalman
2. Find the shape of trajectories:
   i.e. #frames x #trajectories x 2

3. The trajectories are in the following output format:
   frame_id,object_id,x1,y1,w,h,confidence,-1,-1,-1
   Calculate the center from above. Extract a 64x64 patch around that center.

4. After step 3, output should have shape:
   #frames x #trajectories x 64 x 64

5. Then first flatten 64 x 64. Then flatten: #frames x #trajectories

'''
import os
import pandas as pd
import json
from scipy import misc
import numpy as np


def pad_to_square(a, pad_value=1):
  m = a.reshape((a.shape[0], -1))
  padded = pad_value * np.ones(2 * [64], dtype=m.dtype)
  padded[0:m.shape[0], 0:m.shape[1]] = m
  return padded


def center_crop(center_x,center_y,img):
    w, h = 64, 64
    upperleftx = int(center_x - w / 2)
    if (upperleftx < 0):
        upperleftx = 0
    elif (upperleftx >= 1024):
        upperleftx = 1023

    upperlefty = int(center_y - h / 2)
    if (upperlefty < 0):
        upperlefty = 0
    elif (upperlefty >= 1024):
        upperlefty = 1023

    lowerrightx = int(center_x + w / 2)
    if (lowerrightx < 0):
        lowerrightx = 0
    elif (lowerrightx >= 1024):
        lowerrightx = 1023

    lowerrighty = int(center_y + h / 2)
    if (lowerrighty < 0):
        lowerrighty = 0
    elif (lowerrighty >= 1024):
        lowerrighty = 1023

    crop = img[upperlefty:lowerrighty,upperleftx:lowerrightx]
    if (crop.shape != (64,64)):
        crop = pad_to_square(crop)
    return crop



if __name__ == '__main__':
    '''
        if use_patch == True:
            A patch of 64x64 will be cropped around the center and that will be used as input to AR
        else:
            Only center coords: x,y will be used as input to AR
            
    '''
    use_patch = True

    save_path = '/media/narita/Data/Neutrophils/fps_20_frames/6_07072016_Phagocytosis_MRS2578/final' \
                '/tracking/organized_kalman_output_normalized_patch.npy'

    images_path = '/media/narita/Data/Neutrophils/fps_20_frames/6_07072016_Phagocytosis_MRS2578/final/frames/'
    kalman_input_file = '/media/narita/Data/Neutrophils/fps_20_frames/6_07072016_Phagocytosis_MRS2578/final/tracking/' \
                        'kalman_for_gt_labelled_annot.txt'


    #1. Read Kalman Output:
    df = pd.read_table(kalman_input_file, sep=',', header=None)
    df.columns = ["frame_id", "object_id", "x", "y", "w", "h", "conf", "-1", "-1", "-1"]
    object_ids = df["object_id"]

    # this will give the object ids across all frames
    unique_object_ids = set(object_ids)

    '''
    for every object id, extract the its coords, frame_no.
    store it as:
    {
        object_id1:{
                    frame_no_1: (x, y, w, h)
                    frame_no_2: (x, y, w, h)
                  },
        object_id2:{
                    frame_no_1: (x, y, w, h)
                    frame_no_2: (x, y, w, h)
                  },
    
    }
    '''
    final_dict = dict()
    for id in unique_object_ids:
        # extract all rows corresponding to the object_id
        rows = df.loc[df['object_id'] == id]
        new_df = pd.DataFrame(rows)
        current_dict = dict()
        for i in range(len(new_df)):
            frame_id = (new_df.iloc[i, 0])
            x, y, w, h = new_df.iloc[i, 2], new_df.iloc[i, 3], new_df.iloc[i, 4], new_df.iloc[i, 5]
            current_dict[frame_id] = (x, y, w, h)

        #print (len(current_dict.keys()))
        if (len(current_dict.keys()) >= 61):
            final_dict[id] = current_dict

    #print (len(final_dict.keys()))
    #print json.dumps(final_dict,sort_keys = True, indent = 4)
    print ('Length of inner_dict_keys: %d' % (len(final_dict.keys())))

    all_object_crops = []
    for i1,obj in enumerate(final_dict.keys()):
        print (obj)
        inner_dict = final_dict[obj]
        '''
        Sample inner_dict:
                {
                    frame_no_1: (x, y, w, h)
                    frame_no_2: (x, y, w, h)
                }
        '''
        get_img_num = lambda x: (int(x.split('_')[1]))
        inner_dict_keys = sorted(inner_dict.keys(),key=get_img_num)

        if use_patch is True:
            inner_crops_arr = np.empty((len(inner_dict_keys),64,64))
        else:
            inner_crops_arr = np.empty((len(inner_dict_keys), 2))


        for i2,inner_key in enumerate(inner_dict_keys):
            # Read x,y,w,h for a specific object
            # Crop that region from inner_key png
            # Store it in array
            inner_val = inner_dict[inner_key]
            img_name = inner_key

            '''
            scipy.ndimage.imread('img.jpg', mode='RGB'),
            the resulting array will always have this order: (H, W, D) i.e. (height, width, depth) 
            because of the terminology that numpy uses for ndarrays (axis=0, axis=1, axis=2)
            '''
            img = misc.imread(os.path.join(images_path,img_name+'.png'),mode='L')
            mu = 0.4256349084650175
            std = 0.10301448396884938
            img = img/255.
            #img = img - mu
            #img = img / std


            x1,y1,w,h = inner_val

            center_x = (x1 + (x1+w))*0.5
            center_y = (y1 + (y1+h))*0.5
            if use_patch is True:
                center_crop_arr = center_crop(center_x,center_y,img)
                inner_crops_arr[i2]=center_crop_arr
            else:
                inner_crops_arr[i2] = [center_x,center_y]

        print (inner_crops_arr.shape)
        all_object_crops.insert(i1,inner_crops_arr)
        print ('')

    all_object_crops = np.array(all_object_crops)
    print (all_object_crops.shape)
    np.save(save_path,all_object_crops)
    print (save_path + ' saved.')