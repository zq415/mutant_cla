import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import pickle
import nibabel as nib
import random
from scipy import ndimage

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader



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
    

def test_with_probability(model, device, test_loader):
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
            outputs = F.softmax(outputs)
            true_predicted_labels.append((labels.numpy(), predicted.cpu().numpy(), outputs.cpu().numpy()[0,0], outputs.cpu().numpy()[0,1]))
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

def compute_saliency_maps(X, y, model, device):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    ##############################################################################
    # Perform a forward and backward pass through the model to compute the gradient 
    # of the correct class score with respect to each input image. You first want 
    # to compute the loss over the correct scores (we'll combine losses across a batch
    # by summing), and then compute the gradients with a backward pass.
    ##############################################################################
    scores = model(X)
    
    # Get the correct class computed scores.
    scores = scores.gather(1, y.view(-1, 1)).squeeze()  
    
    # Backward pass, need to supply initial gradients of same tensor shape as scores.
    scores.backward(torch.tensor(10.0).cuda(device))
    
    # Get gradient for image.
    saliency = X.grad.data
    
    # Convert from 3d to 1d.
    saliency = saliency.abs()
    saliency = saliency.squeeze()
    ##############################################################################
    return saliency


class Rotate(object):
    
    """
    Rotate the image for data augmentation, but prefer original image.
    """
    
    def __init__(self,ori_probability=0.20):
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