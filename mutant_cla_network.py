import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

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
        if self.transform:
            sample = self.transform(sample)
        return sample
    

import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG_net(nn.Module):
    def __init__(self,conv_drop_rate=0.10,linear_drop_rate=0.10):
        super(VGG_net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_bn = nn.BatchNorm3d(12)
        self.conv2 = nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3,stride=1,padding=2, dilation=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_bn = nn.BatchNorm3d(12)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout3d(conv_drop_rate)
        
        self.conv3 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_bn = nn.BatchNorm3d(24)
        self.conv4 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_bn = nn.BatchNorm3d(24)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.dropout2 = nn.Dropout3d(conv_drop_rate)
        
        self.conv5 = nn.Conv3d(in_channels=24, out_channels=48, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5_bn = nn.BatchNorm3d(48)
        self.conv6 = nn.Conv3d(in_channels=48, out_channels=48, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv6_bn = nn.BatchNorm3d(48)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.dropout3 = nn.Dropout3d(conv_drop_rate)
        
        self.conv7 = nn.Conv3d(in_channels=48, out_channels=72, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv7_bn = nn.BatchNorm3d(72)
        self.conv8 = nn.Conv3d(in_channels=72, out_channels=72, kernel_size=3,stride=1, padding=2,dilation=2)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv8_bn = nn.BatchNorm3d(72)
        self.pool4 = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout4 = nn.Dropout3d(conv_drop_rate)
        
        
        self.fc1 = nn.Linear(72, 2)
        
    def forward(self, x):
        x = self.conv1_bn(self.relu1(self.conv1(x)))
        x = self.dropout1(self.pool1(self.conv2_bn(self.relu2(self.conv2(x)))))
        
        x = self.conv3_bn(self.relu3(self.conv3(x)))
        x = self.dropout2(self.pool2(self.conv4_bn(self.relu4(self.conv4(x)))))
        
        x = self.conv5_bn(self.relu5(self.conv5(x)))
        x = self.dropout3(self.pool3(self.conv6_bn(self.relu6(self.conv6(x)))))
        
        x = self.conv7_bn(self.relu7(self.conv7(x)))
        x = self.conv8_bn(self.relu8(self.conv8(x)))
        
        x1 = self.dropout4(self.pool4(x))
#         x2 = self.dropout5(self.pool5(x))
        
        x1 = x1.view(-1, 72)
#         x2 = x2.view(-1, 72)
#         x = torch.cat((x1, x2), 1)
        x = x1
        #x = self.dropout5(self.fc1_bn(F.relu(self.fc1(x))))
        x = self.fc1(x)
        return x

