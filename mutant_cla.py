
# coding: utf-8

# In[1]:


import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy import ndimage



# In[2]:


def read_mutant_txt(path):
    name_list = []
    fo = open(path)
    for line in fo:
        striped_line = line.strip('\n')
        if striped_line != '':
            name_list.append(striped_line)
    return name_list

def find_centroid(bv_img):
    bv_voxel_num = np.sum(bv_img)
    # find x_centroid
    x_centroid = 0
    for i in range(bv_img.shape[0]):
        x_centroid += np.sum(bv_img[i,:,:])*i
    x_centroid /= bv_voxel_num
    # find y_centroid
    y_centroid = 0
    for i in range(bv_img.shape[1]):
        y_centroid += np.sum(bv_img[:,i,:])*i
    y_centroid /= bv_voxel_num
    # find z_centroid
    z_centroid = 0
    for i in range(bv_img.shape[2]):
        z_centroid += np.sum(bv_img[:,:,i])*i
    z_centroid /= bv_voxel_num
    return (round(x_centroid), round(y_centroid), round(z_centroid))


# In[3]:


def even_func(x):
    if x % 2 == 0:
        return x
    else:
        return x+1
    
def padded_minimum_size(label, min_size):
    img_size= np.shape(label)
    x_offset=0
    y_offset=0
    z_offset=0
    x_flag=False
    y_flag=False
    z_flag=False
    if img_size[0]<min_size:
        x_offset=min_size-img_size[0]
        x_flag=True
    if img_size[1]<min_size:
        y_offset=min_size-img_size[1]
        y_flag=True
    if img_size[2]<min_size:
        z_offset=min_size-img_size[2]
        z_flag=True
    
    if x_flag == True and y_flag == True and z_flag == True:
        padded_label=np.zeros((min_size,min_size,min_size),np.uint8)
        
        padded_label[round(x_offset/2):round(x_offset/2)+img_size[0], 
                     round(y_offset/2):round(y_offset/2)+img_size[1], 
                     round(z_offset/2):round(z_offset/2)+img_size[2]]=label
        
        return padded_label
    
    elif x_flag == True and y_flag == True:
        padded_label=np.zeros((min_size,min_size,even_func(img_size[2])),np.uint8)
        
        padded_label[round(x_offset/2):round(x_offset/2)+img_size[0],
                     round(y_offset/2):round(y_offset/2)+img_size[1],
                     0:img_size[2]] = label
        return padded_label
    
    elif x_flag == True and z_flag == True:
        padded_label=np.zeros((min_size,even_func(img_size[1]),min_size),np.uint8)
        
        padded_label[round(x_offset/2):round(x_offset/2)+img_size[0],
                     0:img_size[1],
                     round(z_offset/2):round(z_offset/2)+img_size[2]]=label
        return padded_label
    
    elif y_flag == True and z_flag == True:
        padded_label=np.zeros((even_func(img_size[0]),min_size,min_size),np.uint8)
        
        padded_label[0:img_size[0],
                     round(y_offset/2):round(y_offset/2)+img_size[1],
                     round(z_offset/2):round(z_offset/2)+img_size[2]]=label
        return padded_label
    
    elif x_flag == True:
        padded_label=np.zeros((min_size,even_fuc(img_size[1]),even_func(img_size[2])),np.uint8)
      
        padded_label[round(x_offset/2):round(x_offset/2)+img_size[0],
                     0:img_size[1],
                     0:img_size[2]]=label
        return padded_label
    
    elif y_flag == True:
        padded_label=np.zeros((even_func(img_size[0]),min_size,even_func(img_size[2])),np.uint8)
       
        padded_label[0:img_size[0],
                     round(y_offset/2):round(y_offset/2)+img_size[1],
                     0:img_size[2]]=label
        return padded_label
    
    elif z_flag == True:
        padded_label=np.zeros((even_func(img_size[0]),even_func(img_size[1]),min_size),np.uint8)

        padded_label[0:img_size[0],
                     0:img_size[1],
                     round(z_offset/2):round(z_offset/2)+img_size[2]]=label
        return padded_label
    
    else:
        padded_label=np.zeros((even_func(img_size[0]),even_func(img_size[1]),even_func(img_size[2])),np.uint8)
       
        padded_label[0:img_size[0],
                     0:img_size[1],
                     0:img_size[2]]=label
        return padded_label

def extract_positive(whole_label,box_size,smallest_ratio,step_size,mul_label,img_idx):
    img_size=np.shape(whole_label)
    bv_voxel_num=np.sum(whole_label)
    positive_sub_volumes=[]
    x_slice,y_slice,z_slice = ndimage.find_objects(whole_label)[0]
    offset=5
    
    x_start = x_slice.stop-offset if x_slice.stop-offset >= box_size else box_size
    x_stop = x_slice.start+box_size+offset if x_slice.start+box_size+offset <= img_size[0] else img_size[0]
    
    y_start = y_slice.stop-offset if y_slice.stop-offset >= box_size else box_size
    y_stop = y_slice.start+box_size+offset if y_slice.start+box_size+offset <= img_size[1] else img_size[1]
    
    z_start = z_slice.stop-offset if z_slice.stop-offset >= box_size else box_size
    z_stop = z_slice.start+box_size+offset if z_slice.start+box_size+offset <= img_size[2] else img_size[2]
    
    for i in range(x_start,x_stop+1,step_size):
        for j in range(y_start,y_stop+1,step_size):
            for k in range(z_start,z_stop+1,step_size):
                contain_ratio = np.sum(whole_label[i-box_size:i,
                                                   j-box_size:j,
                                                   k-box_size:k])/(bv_voxel_num+0.001)
            
                if contain_ratio > smallest_ratio:
                    positive_sub_volumes.append((img_idx,mul_label,(i,j,k)))
    return positive_sub_volumes


# In[4]:


mutant_names = read_mutant_txt('mutant_imgs.txt')
data_base_path = '/scratch/zq415/grammar_cor/Localization/data'
data_folder_list = ['20180419_newdata_nii_with_filtered', 'new_data_20180522_nii', 'organized_data_nii']


# In[5]:


all_BVs = []
for cur_floder in data_folder_list:
    cur_folder_path = os.path.join(data_base_path,cur_floder)
    all_BVs += glob.glob(cur_folder_path+'/*/*/*[Bb][Vv]*')
print(len(all_BVs))


# In[6]:


all_data_dic = {}
same_name_num = 0
for full_bv_path in all_BVs:
    bv_file_name = os.path.basename(full_bv_path)
    if 'BV' in bv_file_name:
        if bv_file_name[:-14] in all_data_dic:
            same_name_num += 1
            continue
        all_data_dic[bv_file_name[:-14]] = full_bv_path
    else:
        if bv_file_name[:-9] in all_data_dic:
            same_name_num += 1
            continue
        all_data_dic[bv_file_name[:-9]] = full_bv_path


# In[ ]:


all_train_data = []
all_idx = []
mutant_list = []
normal_list = []
img_num = 0
normal_num = 0
mutant_num = 0
for key in all_data_dic:
    bv_label = nib.load(all_data_dic[key])
    bv_label = padded_minimum_size(bv_label.get_data(),128)
    #bv_label=np.round(zoom(bv_label,1/2))
    #bv_centroid = find_centroid(bv_label)
    all_train_data.append(bv_label)
    if key in mutant_names:
#def extract_positive(whole_label,box_size,smallest_ratio,step_size,mul_label,img_idx):
        all_idx += extract_positive(bv_label,128,0.96,2,0,img_num)
        mutant_list.append(img_num)
        mutant_num += 1
        img_num += 1
    else:
        all_idx += extract_positive(bv_label,128,0.99,3,1,img_num)
        normal_list.append(img_num)
        normal_num += 1
        img_num += 1
    assert(len(all_train_data)==img_num)
print('mutant number: ', mutant_num)
print('normal number: ', normal_num)
print('train sample number: ', len(all_idx))


# In[ ]:


import pickle
all_data = [all_train_data, all_idx, mutant_list, normal_list]
save_name = 'All_train_bv_data_128.pickle'
save_file = open(os.path.join(os.getcwd(),'data',save_name),'wb')
pickle.dump(all_data,save_file)
save_file.close()


# In[ ]:


from torch.utils.data import Dataset, DataLoader

class Mouse_sub_volumes(Dataset):
    """Mouse sub-volumes BV dataset."""

    def __init__(self, all_whole_volumes, all_idx, transform=None):
        """
        Args:
            all_whole_volumes: Contain all the padded whole BV volumes as a dic
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.whole_volumes = all_whole_volumes
        self.idx = all_idx
        self.transform = transform
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, num):
        #idx [0] dictionary index, [1] label(0 or 1), [2] x, y, z sub-volumes index
        box_size=64
        current_img = self.idx[num][0]
        label = self.idx[num][1]
        x, y, z = self.idx[num][2]
        img = self.whole_volumes[current_img][x-box_size:x,
                                             y-box_size:y,
                                             z-box_size:z]
        img = img[np.newaxis,...]
        sample = {'image': img, 'label': label}
        return sample


# In[ ]:


class Rotate(object):
    
    """
    Rotate the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.30):
        self.ori_probability = ori_probability
        #1:(0,0,0), 2:(90,0,0), 3:(180,0,0), 4:(270,0,0), 5:(0,90,0), 6:(0,270,0)
        self.face_to_you = [1,2,3,4,5,6]
        #rotate along z axis, so that we 24 combination totally
        # 1:0 degree, 2: 90 degree, 3: 180 degree, 4: 270 degree
        self.rotate_z = [1,2,3,4]

    def __call__(self, sample):
        if random.uniform(0,1) < self.ori_probability:
            return sample
        else:
            img, label = sample['image'], sample['label']
            random_choise1=random.choice(self.face_to_you)
            rotated_img1 = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(1,2),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(1,2),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(1,2),reshape='True',mode = 'nearest'),
                            5: lambda x: ndimage.rotate(x,90,(0,2),reshape='True',mode = 'nearest'),
                            6: lambda x: ndimage.rotate(x,270,(0,2),reshape='True',mode = 'nearest')
                            }[random_choise1](img[0,...])
            random_choise2=random.choice(self.rotate_z)
            img[0,...] = {1: lambda x: x,
                            2: lambda x: ndimage.rotate(x,90,(0,1),reshape='True',mode = 'nearest'),
                            3: lambda x: ndimage.rotate(x,180,(0,1),reshape='True',mode = 'nearest'),
                            4: lambda x: ndimage.rotate(x,270,(0,1),reshape='True',mode = 'nearest')
                            }[random_choise2](rotated_img1)
                
        return {'image': img, 'label': label}

class Flip(object):
    
    """
    Flip the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.30):
        self.ori_probability = ori_probability

    def __call__(self, sample):
        if random.uniform(0,1) < self.ori_probability:
            return sample
        else:
            img, label = sample['image'], sample['label']
            random_choise1=random.choice([1,2,3,4,5,6,7,8])
            img[0,...] = {1: lambda x: x,
                          2: lambda x: x[::-1,:,:],
                          3: lambda x: x[:,::-1,:],
                          4: lambda x: x[:,:,::-1],
                          5: lambda x: x[::-1,::-1,:],
                          6: lambda x: x[::-1,:,::-1],
                          7: lambda x: x[:,::-1,::-1],
                          8: lambda x: x[::-1,::-1,::-1]
                          }[random_choise1](img[0,...])
        return {'image': img, 'label': label}

