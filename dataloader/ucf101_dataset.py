import csv
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
import warnings
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import glob

class DataSet(torch.utils.data.Dataset):
    
    
    def __init__(self,root_path,train_or_test,data_list_name='data_file.csv',nb_frames=20,transform=None,concat=False):
        """
        Constructor
        
        Argument:
        root_path: the path to data folder
        train_or_test: True for train data, False for test data
        data_list_name: the filename of data_list
        nb_frames: number of frames extracted from each video
        transform: transform function on PIL image if None, TF.to_tensor is used
        concat: if True, the channel dimension of the tensors from images of frames will be concatenated,
                if False, a 5D tensor [batch_size,channel_size,height,width,sequence_len] will be returned
    
    
        """
        super(DataSet,self).__init__()
        data_list_path = os.path.join(root_path,data_list_name)
        with open(data_list_path,'r') as fin:
            reader = csv.reader(fin)
            self.data_list = list(reader)
            self.data_list = [data for data in self.data_list if len(data)==4 and data[0]==train_or_test]
        self.root_path = root_path
        self.nb_frames = nb_frames
        self.transform = transform
        if self.transform == None:
            self.transform = TF.to_tensor
        self.concat = concat
        
        
        labels = [data[1] for data in self.data_list]
        label_encoder = LabelEncoder()
        labels_index = label_encoder.fit_transform(labels).reshape(-1,1)
        onehot_encoder = OneHotEncoder().fit(labels_index)
        self.label_encoder = label_encoder
        self.classes = self.label_encoder.classes_
        self.onehot_encoder = onehot_encoder
        self.class_transform = lambda x: torch.from_numpy(self.label_encoder.transform(x).flatten())
        
        
    def __len__(self):
        return len(self.data_list)
    
    def get_frames_by_index(self,index):
        data = self.data_list[index]
        label = data[1]
        path = os.path.join(self.root_path, data[0], data[1])
        filename = data[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        if len(images) != self.nb_frames:
            warnings.warn("{0:s} has {1:d} frames".format(filename,len(images)))
            if(len(images))>self.nb_frames:
                images = images[:self.nb_frames]
            else:
                num = self.nb_frames - len(images)
                images.extend([images]*num)
        images = [Image.open(img) for img in images] 
        return images,label
    
    def plot_frames_by_index(self,index):
        
        imgs,label = self.get_frames_by_index(index)
        print(label)
        seq = range(0,self.nb_frames,int(self.nb_frames/4))
        for i,idx in enumerate(seq):
            ax = plt.subplot(1, len(seq), i + 1)
            ax.set_title('#{0:d}'.format(idx))
            plt.imshow(imgs[idx])
    
    def __getitem__(self,index):
        
        imgs,label = self.get_frames_by_index(index)
        imgs = [self.transform(img) for img in imgs]
        if self.concat:
            imgs = torch.cat(imgs,dim=0)
        else:
            shape = list(imgs[0].shape)
            shape.append(1)
            imgs = [img.view(shape) for img in imgs]
            imgs = torch.cat(imgs,dim=3)
        label = self.class_transform([label])
        return imgs,label