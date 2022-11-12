#***************************************************#
#ScriptName: mydataset.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-16 19:49
#Modify Author: fancangning@gmail.com
#Modify Date: 2022-05-16 19:49
#Function: Include a specific class ImageFolder
#***************************************************#
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset_nolist(image_list):
    with open(image_list) as f:
        # list of img path
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, transform=None, target_transform=None, return_paths=False, loader=default_loader, train=False, return_id=False):
        # imgs: list of img path
        imgs, labels = make_dataset_nolist(image_list)
        self.imgs = imgs
        self.labels= labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.return_id = return_id
        self.train = train
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path
        elif self.return_id:
            return img, target ,index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)