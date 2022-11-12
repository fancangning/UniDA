#***************************************************#
#ScriptName: Pretrain.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-17 21:23
#Modify Authot: fancangning@gmail.com
#Modify Date: 2022-05-17 21:23
#Function: Train the model with only source data, therefore offer the initial Mahalanobis detector
#***************************************************#
import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import models
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
from  data_loader.mydataset import ImageFolder
import argparse

parser = argparse.ArgumentParser(description='pretrain the model with only source data')
parser.add_argument('--source_data', type=str, default='./txt/visda/source_train_opda.txt', help='path to source list')
parser.add_argument('--target_data', type=str, default='./txt/visda/target_validation_pda.txt', help='path to target list')
parser.add_argument('--network', type=str, default='resnet50', help='network name')
parser.add_argument('--bottleneck_dim', type=int, default=256, help='the dimension of bottleneck')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of SGD')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--gpu_devices', type=int, nargs='+', default=0, help='which gpu to use')
parser.add_argument('--output_path', type=str, default='snapshot', help='output path')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices


def validate(loader, model):
    start_validate = True
    model.eval()
    with torch.no_grad():
        for img, label in loader:
            img = img.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))
            y, f = model(img)
            if start_validate:
                all_y = y.float().cpu()
                all_label = label.float()
                start_validate = False
            else:
                all_y = torch.cat((all_y, y.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)
    _, predict = torch.max(all_y, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label.cpu()).item() / float(all_label.size()[0])
    return acc


def train():
    # set the log
    output_path = os.path.join(args.output_path, args.source_data.split('_')[1]+'_gpu_'+gpu_devices)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = open(os.path.join(output_path, 'log.txt'), 'w')
    log_file.write(str(vars(args))+'\n')
    log_file.flush()

    # set the datasets
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_folder = ImageFolder(args.source_data, transform=data_transform)
    eval_folder = ImageFolder(args.target_data, transform=eval_data_transform)

    freq = Counter(source_folder.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_folder.labels))
    source_loader = torch.utils.data.DataLoader(source_folder, batch_size=32, sampler=sampler, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(eval_folder, batch_size=32, num_workers=4)

    # set the models
    G = models.resnet.resnet50(pretrained=True)

    bottleneck = nn.Sequential(
        # nn.AdaptativeAvgPool2d(output_size=(1, 1)),
        # nn.Flatten()
        nn.Linear(G.out_features, args.bottleneck_dim),
        nn.BatchNorm1d(args.bottleneck_dim),
        nn.ReLU()
    )

    C = models.basenet.Classifier(G, len(class_weight), bottleneck, args.bottleneck_dim).to(torch.device('cuda'))

    all_parameters = C.get_parameters()
    optimizer = SGD(all_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + 0.001 * float(x)) ** (-0.75))

    # set the loss function
    criterion = nn.CrossEntropyLoss().to(torch.device('cuda'))

    # train
    for epoch in range(args.num_epochs):
        C.train()
        for img_s, label_s in source_loader:
            img_s = img_s.to(torch.device('cuda'))
            label_s = label_s.to(torch.device('cuda'))
            y, f = C(img_s)
            loss = criterion(y, label_s)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        
        acc = validate(test_loader, C)
        log_file.write('epoch: {:05d}, precision: {:.5f}\n'.format(epoch, acc))
        log_file.flush()
        if epoch % 10 == 9:
            torch.save(C.state_dict(), os.path.join(output_path, 'epoch_{:05d}_model.pth.tar'.format(epoch)))
    log_file.close()


def main():
    train()


if __name__ == '__main__':
    main()
