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

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# In[2]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_mutant_txt(path):
    name_list = []
    fo = open(path)
    for line in fo:
        striped_line = line.strip('\n')
        if striped_line != '':
            name_list.append(striped_line)
    return name_list


# In[3]:


mutant_names = read_mutant_txt('mutant_imgs.txt')
data_base_path = '/scratch/zq415/grammar_cor/mutant_detect/data'
data_folder_list = ['20180419_newdata_nii_with_filtered', 'new_data_20180522_nii', 'organized_data_nii']


# In[4]:


save_name = 'All_data_112_64_64.pickle'
with open(os.path.join(os.getcwd(),'data',save_name), "rb") as input_file:
    all_train_data = pickle.load(input_file)


# In[8]:


mutant_group = [(9,10), (12,13,14), (16,17,18,19), (36,37), (38,39), (42,43), (44,45,46,47), (48,49,50,51), (52,53), (54,55),
(56,57,58), (59,60,61), (63,64), (66,67,68), (69,70,71), (73,74), (75,76), (77,78,79), (80,81,82,83,84,85,86,87),
(89,90), (91,92), (93,94), (95,96,97), (98,99), (100,101,102)]

group_list = []
for one_group in mutant_group:
    for ii in range(len(one_group)):
        group_list.append(one_group[ii])
single_mutant = [i for i in range(len(mutant_names)) if i not in group_list]

test_mut_names = []
for i in range(1,len(mutant_group),6):
    for ii in range(len(mutant_group[i])):
        test_mut_names.append(mutant_names[mutant_group[i][ii]])

for i in range(1,len(single_mutant),6):
    test_mut_names.append(mutant_names[single_mutant[i]])
    
print(len(test_mut_names),test_mut_names)


# In[11]:


train_data = []
test_data = []

for i in range(len(all_train_data)):
    if all_train_data[i][2] in mutant_names:
        if all_train_data[i][2] in test_mut_names:
            test_data.append((all_train_data[i][3]-0.5, 0 ))
        else:
            train_data.append((all_train_data[i][3]-0.5,0))
    else:
        random.seed(i*8)
        if random.uniform(0,1) < 0.16:
            test_data.append((all_train_data[i][3]-0.5,1))
        else:
            train_data.append((all_train_data[i][3]-0.5,1))

print(len(test_data))
print(len(train_data))


# In[12]:


#img_nft = nib.Nifti1Image(np.squeeze(all_train_data[193]),np.eye(4))
#img_save_data_path = './img/mul_img.nii'
#nib.save(img_nft,img_save_data_path)


# In[13]:


from torch.utils.data import Dataset, DataLoader

class Mouse_sub_volumes(Dataset):
    """Mouse sub-volumes BV dataset."""

    def __init__(self, all_data , transform=None):
        """
        Args:
            all_whole_volumes: Contain all the padded whole BV volumes as a dic
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_data = all_data
        self.transform = transform
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, num):
        
        current_img, label = self.all_data[num]
        
        img = np.float32(current_img[np.newaxis,...])
        sample = {'image': img, 'label': label}
        return sample


# In[14]:


class Flip(object):
    
    """
    Flip the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.20):
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


# In[15]:


import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG_net(nn.Module):
    def __init__(self,conv_drop_rate=0.15,linear_drop_rate=0.4):
        super(VGG_net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv1_bn = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3,stride=1,padding=2, dilation=2)
        self.conv2_bn = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout3d(conv_drop_rate)
        
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=12, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv3_bn = nn.BatchNorm3d(12)
        self.conv4 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv4_bn = nn.BatchNorm3d(12)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.dropout2 = nn.Dropout3d(conv_drop_rate)
        
        self.conv5 = nn.Conv3d(in_channels=12, out_channels=18, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv5_bn = nn.BatchNorm3d(18)
        self.conv6 = nn.Conv3d(in_channels=18, out_channels=18, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv6_bn = nn.BatchNorm3d(18)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.dropout3 = nn.Dropout3d(conv_drop_rate)
        
        self.conv7 = nn.Conv3d(in_channels=18, out_channels=24, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv7_bn = nn.BatchNorm3d(24)
        self.conv8 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3,stride=1, padding=2,dilation=2)
        self.conv8_bn = nn.BatchNorm3d(24)
        self.pool4 = nn.MaxPool3d(2, 2)
        self.dropout4 = nn.Dropout3d(conv_drop_rate)
        
        self.fc1 = nn.Linear(7*4*4*24, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(linear_drop_rate)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool1(self.conv2_bn(F.relu(self.conv2(x)))))
        
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.dropout2(self.pool2(self.conv4_bn(F.relu(self.conv4(x)))))
        
        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.dropout3(self.pool3(self.conv6_bn(F.relu(self.conv6(x)))))
        
        x = self.conv7_bn(F.relu(self.conv7(x)))
        x = self.dropout4(self.pool4(self.conv8_bn(F.relu(self.conv8(x)))))
        
        x = x.view(-1, 7*4*4*24)
        x = self.dropout5(self.fc1_bn(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x


# In[16]:


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(train_loader):
        inputs, labels = sample_batched['image'], sample_batched['label']  
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        i_batch += 1
        if i_batch % 10 == 0:
            print("epoch {}, batch {}, current loss {}".format(epoch+1,i_batch,running_loss/10))
            running_loss = 0.0

def test(model, device, test_loader):
    model.eval()
    correct_num = 0
    total_num = 0
    positive_correct=0
    positive_num=0
    negative_correct=0
    negative_num=0
    
    true_predicted_labels = []
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            inputs, labels = sample_batched['image'], sample_batched['label']  
            inputs = inputs.to(device)
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_predicted_labels.append((labels.numpy(), predicted.cpu().numpy()))
            correct_num+=np.sum(predicted.cpu().numpy()==labels.numpy())
            total_num+=len(labels)
            positive_correct+=np.sum(predicted.cpu().numpy()*labels.numpy())
            positive_num+=np.sum(labels.numpy())
            negative_correct+=np.sum((1-predicted.cpu().numpy())*(1-labels.numpy()))
            negative_num+=np.sum(1-labels.numpy())
            
    print('total_num:{}, test accuracy:{}, positive_acc:{}, negative_acc:{}'.format(total_num,
                                                                                   correct_num/total_num,
                                                                                    positive_correct/positive_num,
                                                                                    negative_correct/negative_num
                                                                                    ))
    return true_predicted_labels

def get_confusion_matrix(true_predicted_labels):
    cross_table = np.zeros([2,2])
    mut_to_nor = []
    nor_to_mul = []
    test_dic = true_predicted_labels
    for i in range(len(test_dic)):
        if test_dic[i][0] ==0 and test_dic[i][1] ==0:
            cross_table[0,0] += 1
        elif  test_dic[i][0] ==0 and test_dic[i][1] ==1:
            cross_table[0,1] += 1
            mut_to_nor.append(i)
        elif test_dic[i][0] ==1 and test_dic[i][1] ==0:
            cross_table[1,0] += 1
            nor_to_mul.append(i)
        elif test_dic[i][0] ==1 and test_dic[i][1] ==1:
            cross_table[1,1] += 1
    print(cross_table)
    print(mut_to_nor)
    print(nor_to_mul)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = VGG_net()
#net.apply(weight_init)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
net.to(device)
print("There are {} parameters in the model".format(count_parameters(net)))

num_epochs = 420
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.0,1.0]).to(device))

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
print('choose SGD as optimizer')
#optimizer = optim.Adam(net.parameters(), lr=args.lr*10, weight_decay=0.00001)
#print('choose Adam as optimizer')

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

for epoch in range(num_epochs):
    scheduler.step()
    
    Mouse_dataset = Mouse_sub_volumes(train_data, transform=transforms.Compose([Flip()]))
    dataloader = DataLoader(Mouse_dataset, batch_size=12, shuffle=True, num_workers=4, drop_last = True)
    train(net, device, dataloader, optimizer, criterion, epoch)
    if (epoch + 1) % 50 == 0:
        print('epoch {} train accuracy: '.format(epoch+1))
        train_Mouse_dataset = Mouse_sub_volumes(train_data)
        train_dataloader = DataLoader(train_Mouse_dataset, batch_size=1, shuffle=False, num_workers=4)
        train_dic = test(net, device, train_dataloader)
        get_confusion_matrix(train_dic)
        
        print("-------------------")
        print('epoch {} test accuracy: '.format(epoch+1))
        test_Mouse_dataset = Mouse_sub_volumes(test_data)
        test_dataloader = DataLoader(test_Mouse_dataset, batch_size=1, shuffle=False, num_workers=4)
        test_dic = test(net, device, test_dataloader)
        get_confusion_matrix(test_dic)
        
        torch.save(net.state_dict(), './model/mut_clas_2019_01_17_e{}.pth'.format(epoch+1))


