#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import nibabel as nib
import os
from sklearn.decomposition import PCA
import math
from skimage import io
import plotly.plotly as py
from plotly.grid_objs import Grid, Column
import time
#get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import rotate as rot
import glob
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import pickle



# In[2]:


def read_mutant_txt(path):
    name_list = []
    fo = open(path)
    for line in fo:
        striped_line = line.strip('\n')
        if striped_line != '':
            name_list.append(striped_line)
    return name_list

def l2norm(a,b):
    return np.sqrt(a**2+b**2)


def align_img(bv_label):
    x,y,z = np.where(bv_label==1)
    points = np.vstack((x,y,z))
    points = points.T
    #print(points.shape)
    pca = PCA(n_components = 3)
    pca.fit(points)
    pc1 = pca.components_[0,:]
    #print(pc1)
    pc2 = pca.components_[1,:]
    if pc1[0] >=0 and pc1[1] >=0:
        azimuth = -np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi))
    #print('case1')
#     clockwise = 1

    elif pc1[0] <=0 and pc1[1] <=0:
        azimuth = (180 - np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi)))
    #    print('case2')
    #     anticlockwise = 1

    elif pc1[0]>=0 and pc1[1]<=0:
        azimuth = np.arctan(np.abs(pc1[1])/np.abs(pc1[0]))*(360/(2*np.pi))
    #    print('case3')
    #     anticlockwise = 1
    else:
        azimuth = -(90 + np.arctan(np.abs(pc1[0])/np.abs(pc1[1]))*(360/(2*np.pi)))
    #    print('case4')
    #     clockwise = 1

    # finding the elevation angle
    if pc1[2]>=0:
        elevation = -np.arctan(pc1[2]/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
     #   print('case1')
    else:
        elevation = np.arctan(np.abs(pc1[2])/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
     #   print('case2')
   
    bv_label_copy = bv_label.copy()
    label_rot1 = rot(bv_label_copy,angle=azimuth,axes=(0,1))
    label_rot = rot(label_rot1,angle=elevation,axes=(0,2))
    
    xr,yr,zr = np.where(label_rot==1)
    points_rot = np.vstack((xr,yr,zr))
    points_rot = points_rot.T
    #print(points_rot.shape)
    pca_rot = PCA(n_components = 3)
    pca_rot.fit(points_rot)
    pc1r = pca_rot.components_[0,:]
    pc2r = pca_rot.components_[1,:]
    
    angle3 = np.arctan(pc2r[2]/pc2r[1])*(360/(2*np.pi))
    
    label_rot = rot(label_rot,angle=-angle3,axes=(1,2))
    
    return azimuth, elevation, angle3, label_rot

def save_img(img, label, count):
    img_nft = nib.Nifti1Image(img,np.eye(4))
    img_save_data_path = './resize_img/img{}.nii'.format(count)
    nib.save(img_nft,img_save_data_path)
    
    img_nft = nib.Nifti1Image(label,np.eye(4))
    img_save_data_path = './resize_img/label{}.nii'.format(count)
    nib.save(img_nft,img_save_data_path)
    


# In[3]:


data_base_path = '/scratch/zq415/grammar_cor/Localization/data'
data_folder_list = ['20180419_newdata_nii_with_filtered', 'new_data_20180522_nii']
data_folder_list2 = 'fix_organized_data_nii'

all_BVs = []
for cur_floder in data_folder_list:
    cur_folder_path = os.path.join(data_base_path,cur_floder)
    all_BVs += glob.glob(cur_folder_path+'/*/*/*[Bb][Vv]*')
print(len(all_BVs))

cur_folder_path = os.path.join(data_base_path,data_folder_list2)
all_BVs += glob.glob(cur_folder_path+'/*[Bb][Vv]*')
print(len(all_BVs))

all_data_list = []
same_name_num = 0
for full_bv_path in all_BVs:
    if 'BV' in full_bv_path:
        all_data_list.append((full_bv_path[:-14] + '.nii', full_bv_path))
    else:
        all_data_list.append((full_bv_path[:-9]+ '_2' + '.nii', full_bv_path))


# In[4]:


mutant_names = read_mutant_txt('mutant_imgs.txt')


# In[5]:


count = 0
mutant_label = {}

for i,img_path in enumerate(all_data_list):
    img = nib.load(img_path[0])
    img = np.float32(img.get_data())
    
    img_label = nib.load(img_path[1])
    img_label = np.uint8(img_label.get_data())
    img_label[img_label>0] = 1
    
    print(np.shape(img),np.shape(img_label))
    
    azimuth, elevation, angle3, label_rot = align_img(img_label)
    
    img_rot1 = rot(img,angle=azimuth,axes=(0,1))
    img_rot2 = rot(img_rot1,angle=elevation,axes=(0,2))
    img_rot = rot(img_rot2,angle=-angle3,axes=(1,2))
    
    x_slice,y_slice,z_slice = ndimage.find_objects(label_rot)[0]
    print(count, 'bv size: ', x_slice.stop-x_slice.start, y_slice.stop-y_slice.start, z_slice.stop-z_slice.start)
    
    img_slice = img_rot[x_slice.start:x_slice.stop, y_slice.start:y_slice.stop, z_slice.start:z_slice.stop]
    label_slice = label_rot[x_slice.start:x_slice.stop, y_slice.start:y_slice.stop, z_slice.start:z_slice.stop]
    
    x, y, z = np.shape(label_slice)
    label_resized = zoom(label_slice, (112.0/x, 64.0/y, 64.0/z))
    label_resized[label_resized>=0.5] = 1
    label_resized[label_resized<0.5] = 0
    
    img_resized = zoom(img_slice, (112.0/x, 64.0/y, 64.0/z))
    
    save_img(img_resized, label_resized, count)
    count += 1
    
    if 'BV' in img_path[1]:
        bv_base_name = os.path.basename(img_path[0])[:-4]
    else:
        bv_base_name = os.path.basename(img_path[0])[:-6]
        
    if bv_base_name in mutant_names:
        mutant_label[i] = (i, 0, bv_base_name, label_resized, img_resized, img_path[1])
    else:
        mutant_label[i] = (i, 1, bv_base_name, label_resized, img_resized, img_path[1])
        


# In[6]:


save_name = 'All_data_112_64_64.pickle'
save_file = open(os.path.join(os.getcwd(),'data',save_name),'wb')
pickle.dump(mutant_label,save_file)
save_file.close()


