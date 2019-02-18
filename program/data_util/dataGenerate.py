import os

import numpy as np
import skimage.io
import itertools
import PIL.Image as Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

def load_image(path):
        image_red_ch = skimage.io.imread(path+'_red.png')
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = skimage.io.imread(path+'_green.png')
        image_blue_ch = skimage.io.imread(path+'_blue.png')
        

        image_red_ch += (image_yellow_ch/2).astype(np.uint8) 
        image_green_ch += (image_yellow_ch/2).astype(np.uint8)

        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch), -1)
        
        return image


def label_decoder(label,class_num=28):
    label_list = label.split(" ")
    label_decoder_list = list(np.zeros(class_num,dtype=np.float32))
    for i in range(class_num):
        if str(i) in label_list:
            label_decoder_list[i]=1.0
    return np.asarray(label_decoder_list,dtype=np.float32)


class DatasetGenerate(Dataset):
    def __init__(self, path,label_list, transform=None, target_transform=None, loader=load_image,decoder=label_decoder):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((os.path.join(path,row['Id']), row['Target']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.decoder = label_decoder

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        img = Image.fromarray(img,'RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.decoder(label)
        return img, label

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, path,label_list, transform=None, target_transform=None, loader=load_image):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((os.path.join(path,row['Id'])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = self.loader(filename)
        img = Image.fromarray(img,'RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


def test_data_generate(path,data_list):
    train_dataset = DatasetGenerate(path,data_list)
    fig, ax = plt.subplots(1,5,figsize=(25,5))
    for i in range(5):
        ax[i].imshow(train_dataset.__getitem__(i)[0])