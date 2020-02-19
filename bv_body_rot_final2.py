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
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import rotate as rot
from os.path import split
from scipy.spatial.distance import euclidean as edist
import glob
from skimage import measure
import copy
import cv2


# In[2]:


root_dir_path = '/scratch/zq415/grammar_cor/bv_body/dataset/nii_train'
save_path = '/scratch/zq415/grammar_cor/mutant_detect/mutant_cla/rotate_bv'

files = glob.glob(root_dir_path + '/*/*.nii')
all_img_file_path = [file for file in files if 'BV' not in file and 'prebv' not in file and 'BODY'not in file]

print(len(all_img_file_path))


# In[3]:


def l2norm(a,b):
    return np.sqrt(a**2+b**2)

def region_cen(region):
    x,y,z = np.where(region == 2)
    if x.shape[0] == 0:
        return np.array([-1,-1,-1])
    x_mean = np.sum(x)/x.shape[0]
    y_mean = np.sum(y)/y.shape[0]
    z_mean = np.sum(z)/z.shape[0]
    region_center = np.array([x_mean,y_mean,z_mean])
    return region_center

def get_sum_components(region):
    region[region==1] = 0  
    region[region==2] = 1
    total_num_component = 0
    for i in range(0, region.shape[1], 1):
        kernel = np.ones((3,3),np.uint8)
        bv_slide = cv2.dilate(region[:,i,:],kernel,iterations = 1)
        bv_slide = cv2.erode(bv_slide,kernel,iterations = 1)
        bv_slide_component = measure.label(bv_slide)
        total_num_component += np.max(bv_slide_component)
    return total_num_component
    
    
# def rotate(label):
#     # for body
#     x,y,z = np.where(label >= 1 )
#     points = np.vstack((x,y,z))
#     points = points.T
#     pca = PCA(n_components = 3)
#     pca.fit(points)
#     pc1_body = pca.components_[0,:]
#     pc2_body = pca.components_[1,:]
#     pc3_body = pca.components_[2,:]
    
#     # for bv
#     x,y,z = np.where(label == 2 )
#     points = np.vstack((x,y,z))
#     points = points.T
#     pca.fit(points)
#     pc1_bv = pca.components_[0,:]
#     pc2_bv = pca.components_[1,:]
#     pc3_bv = pca.components_[2,:]

#     pc1 = pc1_body

#     pc2 = pc1_bv
#     #  finding angle 1
#     # projection of PC1 on the xy plane
#     # clockwise = 0
#     # anticlockwise = 0
#     if pc1[0] >=0 and pc1[1] >=0:
#         azimuth = -np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi))
#         print('case1')

#     elif pc1[0] <=0 and pc1[1] <=0:
#         azimuth = (180 - np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi)))
#         print('case2')

#     elif pc1[0]>=0 and pc1[1]<=0:
#         azimuth = np.arctan(np.abs(pc1[1])/np.abs(pc1[0]))*(360/(2*np.pi))
#         print('case3')
    
#     else:
#         azimuth = -(90 + np.arctan(np.abs(pc1[0])/np.abs(pc1[1]))*(360/(2*np.pi)))
#         print('case4')

#     # finding the elevation angle
#     if pc1[2]>=0:
#         elevation = -np.arctan(pc1[2]/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
#         print('case1')
#     else:
#         elevation = np.arctan(np.abs(pc1[2])/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
#         print('case2')
    
#     print('azimuth={}, elevation={}'.format(azimuth,elevation))
#     label_copy = label.copy()
#     label_rot1 = rot(label_copy,angle=azimuth,axes=(0,1))
#     label_rot = rot(label_rot1,angle=elevation,axes=(0,2))
    
#     #### PCA second time
#     xr,yr,zr = np.where(label_rot>=.9)
#     points_rot = np.vstack((xr,yr,zr))
#     points_rot = points_rot.T
#     pca_rot = PCA(n_components = 3)
#     pca_rot.fit(points_rot)
#     pc1r = pca_rot.components_[0,:]
#     pc2r = pca_rot.components_[1,:]
    
#     # calculating angle 3 and rotating with the third angle
#     angle3 = np.arctan(pc2r[2]/pc2r[1])*(360/(2*np.pi))
#     print('angle3=',angle3)
#     label_rot = rot(label_rot,angle=-angle3,axes=(1,2))
    
#     label_rot = np.round(label_rot)
    
#     # finding the bv and body centers to fix orientation
#     x,y,z = np.where(label_rot >= 1)
#     x_mean_body = np.sum(x)/x.shape[0]
#     y_mean_body = np.sum(y)/y.shape[0]
#     z_mean_body = np.sum(z)/z.shape[0]
#     body_center = (x_mean_body,y_mean_body,z_mean_body)
#     x,y,z = np.where(label_rot == 2)
#     x_mean_bv = np.sum(x)/x.shape[0]
#     y_mean_bv = np.sum(y)/y.shape[0]
#     z_mean_bv = np.sum(z)/z.shape[0]
#     bv_center = (x_mean_bv,y_mean_bv,z_mean_bv)
#     if bv_center[0] < body_center[0]:
#         label_rot = rot(label_rot,angle=180,axes=(0,1))
# #         img_rot = rot(img_rot,angle=180,axes=(0,1))
#     # for generating the final rotation
#     x,y,z = np.where(label_rot == 2)
#     x_mean_bv = np.sum(x)/x.shape[0]
#     y_mean_bv = np.sum(y)/y.shape[0]
#     z_mean_bv = np.sum(z)/z.shape[0]
#     bv_center = (x_mean_bv,y_mean_bv,z_mean_bv)
    
#     lv_region = label_rot[:,0:int(bv_center[1]),0:int(bv_center[2])]
#     rv_region = label_rot[:,0:int(bv_center[1]),int(bv_center[2])+1:]
#     lback_region = label_rot[:,int(bv_center[1]):,0:int(bv_center[2])]
#     rback_region = label_rot[:,int(bv_center[1]):,int(bv_center[2])+1:]
#     lv_region_cen = region_cen(lv_region) 
#     rv_region_cen = region_cen(rv_region) + np.array([0,0,int(bv_center[2])])
#     lback_region_cen = region_cen(lback_region) + np.array([0,int(bv_center[1]),0])
#     rback_region_cen = region_cen(rback_region) + np.array([0,int(bv_center[1]),int(bv_center[2])])

#     if -1 in lv_region_cen or -1 in rv_region_cen or -1 in lback_region_cen or -1 in rback_region_cen:
#         print('-1,-1,-1 returned')
#         return label_rot

#     d1 = edist(lv_region_cen,rv_region_cen)
#     d2 = edist(lback_region_cen,rback_region_cen)
    
#     if d1 < d2 :
# #         print('d1:',d1,'\t <','d2',d2)
#         print('performing final flip')
#         label_rot = rot(label_rot,angle=180,axes=(1,2))
# #         img_rot = rot(img_rot,angle=180,axes=(1,2))
        
# #     return img_rot , label_rot,lv_region,rv_region,lback_region,rback_region
#     return label_rot
    
    
def rotate2(label):
    # for body
    x,y,z = np.where(label >= 1 )
    points = np.vstack((x,y,z))
    points = points.T
    pca = PCA(n_components = 3)
    pca.fit(points)
    pc1_body = pca.components_[0,:]
    pc2_body = pca.components_[1,:]
    pc3_body = pca.components_[2,:]
    
    # for bv
    x,y,z = np.where(label == 2 )
    points = np.vstack((x,y,z))
    points = points.T
    pca.fit(points)
    pc1_bv = pca.components_[0,:]
    pc2_bv = pca.components_[1,:]
    pc3_bv = pca.components_[2,:]

    pc1 = pc1_body

    pc2 = pc1_bv

    if pc1[0] >=0 and pc1[1] >=0:
        azimuth = -np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi))
        print('case1')

    elif pc1[0] <=0 and pc1[1] <=0:
        azimuth = (180 - np.arctan(pc1[1]/pc1[0])*(360/(2*np.pi)))
        print('case2')

    elif pc1[0]>=0 and pc1[1]<=0:
        azimuth = np.arctan(np.abs(pc1[1])/np.abs(pc1[0]))*(360/(2*np.pi))
        print('case3')
    
    else:
        azimuth = -(90 + np.arctan(np.abs(pc1[0])/np.abs(pc1[1]))*(360/(2*np.pi)))
        print('case4')

    # finding the elevation angle
    if pc1[2]>=0:
        elevation = -np.arctan(pc1[2]/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
        print('case1')
    else:
        elevation = np.arctan(np.abs(pc1[2])/l2norm(pc1[0],pc1[1]))*(360/(2*np.pi))
        print('case2')
    
    print('azimuth={}, elevation={}'.format(azimuth,elevation))
    label_copy = label.copy()
    label_rot1 = rot(label_copy,angle=azimuth,axes=(0,1))
    label_rot = rot(label_rot1,angle=elevation,axes=(0,2))
    
    #### PCA second time
    xr,yr,zr = np.where(label_rot>=.9)
    points_rot = np.vstack((xr,yr,zr))
    points_rot = points_rot.T
    pca_rot = PCA(n_components = 3)
    pca_rot.fit(points_rot)
    pc1r = pca_rot.components_[0,:]
    pc2r = pca_rot.components_[1,:]
    
    # calculating angle 3 and rotating with the third angle
    angle3 = np.arctan(pc2r[2]/pc2r[1])*(360/(2*np.pi))
    print('angle3=',angle3)
    label_rot = rot(label_rot,angle=-angle3,axes=(1,2))
    
    label_rot = np.round(label_rot)
    
    # finding the bv and body centers to fix orientation
    x,y,z = np.where(label_rot >= 1)
    x_mean_body = np.sum(x)/x.shape[0]
    y_mean_body = np.sum(y)/y.shape[0]
    z_mean_body = np.sum(z)/z.shape[0]
    body_center = (x_mean_body,y_mean_body,z_mean_body)
    x,y,z = np.where(label_rot == 2)
    x_mean_bv = np.sum(x)/x.shape[0]
    y_mean_bv = np.sum(y)/y.shape[0]
    z_mean_bv = np.sum(z)/z.shape[0]
    bv_center = (x_mean_bv,y_mean_bv,z_mean_bv)
    if bv_center[0] < body_center[0]:
        label_rot = rot(label_rot,angle=180,axes=(0,1))
#         img_rot = rot(img_rot,angle=180,axes=(0,1))
    # for generating the final rotation
    x,y,z = np.where(label_rot == 2)
    x_mean_bv = np.sum(x)/x.shape[0]
    y_mean_bv = np.sum(y)/y.shape[0]
    z_mean_bv = np.sum(z)/z.shape[0]
    bv_center = (x_mean_bv,y_mean_bv,z_mean_bv)
    
    bv_front_region = copy.deepcopy(label_rot[:,min(y):int(bv_center[1]),:])
    bv_back_region = copy.deepcopy(label_rot[:,int(bv_center[1]):max(y),:])
    
    c1 = get_sum_components(bv_front_region)
    c2 = get_sum_components(bv_back_region)
    print("c1: ", c1, "c2: ", c2)
    
    if c1 < c2 :
#         print('d1:',d1,'\t <','d2',d2)
        print('performing final flip')
        label_rot = rot(label_rot,angle=180,axes=(1,2))
#         img_rot = rot(img_rot,angle=180,axes=(1,2))
        
#     return img_rot , label_rot,lv_region,rv_region,lback_region,rback_region
    return label_rot


# In[ ]:


count = 1
for img_path in all_img_file_path:
    print('#############',count,'############')
    img_name = os.path.split(img_path)[1]
    label_name = img_name[:-4] + 'bv_body_label.nii'
    print(img_name,label_name)
#     img = np.array(nib.load(img_path).dataobj)
    
    bv_label_path = img_path[:-4] + '_BV_labels.nii'
    if not os.path.exists(bv_label_path):
        bv_label_path = img_path[:-4] + '_prebv.nii'
            
    body_label_path = img_path[:-4] + '_BODY_labels.nii'
        
    bv_label = np.array(nib.load(bv_label_path).dataobj)
    body_label = np.array(nib.load(body_label_path).dataobj)
    
    x_len, y_len, z_len = body_label.shape
    label = bv_label[0:x_len, 0:y_len, 0:z_len] + body_label


#     print('image size:',img.shape)
    print('label size',label.shape)
    # FUNCTION CALL TO ROTATE
#     img_out,label_out,lvr,rvr,lbr,rbr = rotate(label,img)
    label_out = rotate2(label)
    ### SAVING
#     img_rot_nib = nib.Nifti1Image(np.squeeze(img_out),np.eye(4))
    label_rot_nib = nib.Nifti1Image(np.squeeze(np.round(label_out).astype(np.uint8)), np.eye(4))
    
#     nib.save(img_rot_nib, os.path.join(save_path, img_name))
    nib.save(label_rot_nib, os.path.join(save_path, label_name))
    
    count += 1


