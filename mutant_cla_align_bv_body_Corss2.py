#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import pickle
import nibabel as nib
import random
from scipy.ndimage.interpolation import zoom
import copy
import Levenshtein

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchcontrib.optim import SWA

from mutant_cla_network import *
from utility_func import *


# In[2]:


mutant_names = read_mutant_txt('mutant_imgs.txt')
data_base_path = '/scratch/zq415/grammar_cor/mutant_detect/data'
# data_folder_list = ['20180419_newdata_nii_with_filtered', 'new_data_20180522_nii', 'organized_data_nii']


# In[3]:


# mutant_label[i] = (i, 1, bv_base_name, label_resized, img_resized, img_path[1])

save_name = 'All_unalign_bv_body_data_128_128_128_down.pickle'
cross_group_name = 'fold_name_6.pickle'
with open(os.path.join(os.getcwd(),'data',save_name), "rb") as input_file:
    all_train_data = pickle.load(input_file)
    
with open(os.path.join(os.getcwd(),'data',cross_group_name), "rb") as input_file:
    data_folds = pickle.load(input_file)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 400
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.5,1.0]).to(device))


for fold_num in range(6):
    if fold_num <= 3:
        continue

    net = VGG_net()
    #net.apply(weight_init)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    print("There are {} parameters in the model".format(count_parameters(net)))
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
    optimizer = SWA(optimizer, swa_start=280, swa_freq=5)
    print('choose SGD as optimizer')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=190, gamma=0.5)
    
    train_data = []
    test_data = []
    used_name_list = []
    test_mut_num = 0
    train_mut_num = 0
    for current_fold in range(6):
        if current_fold == fold_num:
            for i in range(len(data_folds[current_fold])):
                if data_folds[current_fold][i][0] in all_train_data and data_folds[current_fold][i][0] not in used_name_list:
                    cur_img = np.float32(all_train_data[data_folds[current_fold][i][0]] >= 0.5)
#                     cur_img = all_train_data[data_folds[current_fold][i][0]]
                    test_data.append((cur_img - 0.5, data_folds[current_fold][i][1]))
                    used_name_list.append(data_folds[current_fold][i][0])
                    if data_folds[current_fold][i][1] == 0:
                        test_mut_num += 1
        else:
            for i in range(len(data_folds[current_fold])):
                if data_folds[current_fold][i][0] in all_train_data and data_folds[current_fold][i][0] not in used_name_list:
                    cur_img = np.float32(all_train_data[data_folds[current_fold][i][0]] >= 0.5)
#                     cur_img = all_train_data[data_folds[current_fold][i][0]]
                    train_data.append((cur_img - 0.5, data_folds[current_fold][i][1]))
                    used_name_list.append(data_folds[current_fold][i][0])
                    if data_folds[current_fold][i][1] == 0:
                        train_mut_num += 1
                                 
    print('current fold: ', fold_num, 'test_data_len: ', len(test_data), 'train_data_len: ', len(train_data),
         'total len: ', len(test_data)+len(train_data))
    print('test_mut_num: ', test_mut_num, 'train_mut_num: ', train_mut_num)
                                 
    for epoch in range(num_epochs):
        scheduler.step()

        Mouse_dataset = Mouse_sub_volumes(train_data, transform=transforms.Compose([Rotate(), Flip()]))
        dataloader = DataLoader(Mouse_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last = True)
        train(net, device, dataloader, optimizer, criterion, epoch)
        if (epoch + 1) % 10 == 0:
            print('fold {}, epoch {} train accuracy: '.format(fold_num, epoch+1))
            train_Mouse_dataset = Mouse_sub_volumes(train_data)
            train_dataloader = DataLoader(train_Mouse_dataset, batch_size=1, shuffle=False, num_workers=4)
            train_dic = test(net, device, train_dataloader)
            get_confusion_matrix(train_dic)

            print("-------------------")
            print('fold {}, epoch {} test accuracy: '.format(fold_num, epoch+1))
            test_Mouse_dataset = Mouse_sub_volumes(test_data)
            test_dataloader = DataLoader(test_Mouse_dataset, batch_size=1, shuffle=False, num_workers=4)
            test_dic = test(net, device, test_dataloader)
            get_confusion_matrix(test_dic)

            torch.save(net.state_dict(), './model/mut_clas_2020_03_25_e{}_global3_fold{}_raw_body2.pth'.format(epoch+1,fold_num))

    del net, optimizer, scheduler





