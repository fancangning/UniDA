#***************************************************#
#ScriptName: get_loader.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-16 14:52
#Modify Author: fancangning@gmail.com
#Modify Date: 2022-05-16 14:52
#Function: Include some functions: get_loader
#***************************************************#
from .mydataset import ImageFolder
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32, return_id=False, balanced=False, val=False, val_data=None):
    source_folder = ImageFolder(os.path.join(source_path), transform=transforms[source_path], return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path), transform=transforms[target_path], return_paths=False, return_id=return_id)
    if val:
        source_val_train = ImageFolder(val_data, transforms[source_path], return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path), transform=transforms["eval"], return_paths=True)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights, len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(source_folder, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(source_folder, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    target_loader = torch.utils.data.DataLoader(target_folder_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(eval_folder_test, batch_size=batch_size, shuffle=False, num_workers=4)
    if val:
        test_loader_source = torch.utils.data.DataLoader(source_val_test, batch_size=batch_size, shuffle=False, num_workers=4)
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader, target_loader, test_loader, target_folder_train