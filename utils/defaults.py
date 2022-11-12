#***************************************************#
#ScriptName: defaults.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-16 13:57
#Modify Author: fancangning@gmail.com
#Modify Date: 2022-05-16 13:57
#Function: Include some functions: get_dataloaders(kwargs) get_model()
#***************************************************#
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import models
from data_loader.get_loader import get_loader, get_loader_labels


def get_model(kwargs):
    model_name = kwargs['network']
    num_classes = kwargs['num_classes']
    bottleneck_dim = kwargs['bottleneck_dim']
    randomized = kwargs['randomized']
    randomized_dim = kwargs['randomized_dim']
    pretrain = kwargs['pretrain']
    if model_name in models.resnet.__dict__:
        G = models.resnet.__dict__[model_name](pretrained=pretrain)
    else:
        raise Exception('Invalid model name')
    
    bottleneck = nn.Sequential(
        # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        # nn.Flatten(),
        nn.Linear(G.out_features, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.ReLU()
    )
    C = models.basenet.Classifier(G, num_classes, bottleneck, bottleneck_dim).to(torch.device('cuda'))
    if randomized:
        D = models.basenet.DomainDiscriminator(randomized_dim, hidden_size=1024).to(torch.device('cuda'))
    else:
        D = models.basenet.DomainDiscriminator(C.features_dim * num_classes, hidden_size=1024).to(torch.device('cuda'))
    
    return C, D


def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)