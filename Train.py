#***************************************************#
#ScriptName: Train.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-16 12:26
#Modify Authot: fancangning@gmail.com
#Modify Date: 2022-05-16 12:26
#Function: Train the model
#***************************************************#
import yaml
import easydict
import os
import numpy as np
from models.basenet import ConditionalDomainAdversarialLoss, accuracy
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import models
import lib_generation
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
from data_loader.mydataset import ImageFolder
import argparse

parser = argparse.ArgumentParser(description='Mahalanobis detector for UniDA')
parser.add_argument('--source_data', type=str, default='./txt/source_amazon_opda.txt', help='path to source list')
parser.add_argument('--target_data', type=str, default='./txt/target_webcam_opda.txt', help='path to target list')
parser.add_argument('--network', type=str, default='resnet50', help='network name')
parser.add_argument('--bottleneck_dim', type=int, default=256, help='the dimension of bottleneck')
parser.add_argument('--randomized', default=True, action='store_true')
parser.add_argument('--randomized_dim', type=int, default=1024, help='randomized dimension when using randomized multi-linear-map')
parser.add_argument('--class_num', type=int, default=9, help='the number of source classes')
parser.add_argument('--shared_class_num', type=int, default=6, help='the number of shared classes between source and target domains')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of SGD')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--iters_per_epoch', type=int, default=1000, help='Number of iterations per epoch')
parser.add_argument('--score_thr', type=float, default=-230, help='the threshold of the Mahalanobis score')
parser.add_argument('--second_score_thr', type=float, default=-150, help='the threshold of the Mahalanobis score after the first epoch')
parser.add_argument('--scale', type=float, default=12.0, help='the amplification factor for Mahalanobis score')
parser.add_argument('--trade_off', type=float, default=1., help='the trade-off hyper-parameter for transfer loss')
parser.add_argument('--gpu_devices', type=int, nargs='+', default=0, help='which gpu to use')
parser.add_argument('--save_model', default=False, action='store_true')
parser.add_argument('--pretrain_path', type=str, default='./snapshot/amazon_gpu_0/epoch_00059_model.pth.tar')
parser.add_argument('--output_path', type=str, default='snapshot', help='path to save result')
parser.add_argument('--exp_dir', type=str, default='log', help='dir name to save result')
args = parser.parse_args()

gpu_devices = str(args.gpu_devices[0])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

output_path = os.path.join(args.output_path, args.source_data.split('_')[1]+'_'+args.target_data.split('_')[1]+'_gpu'+gpu_devices, args.exp_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)
log_file = open(os.path.join(output_path, 'log.txt'), 'w')
log_file.write(str(vars(args))+'\n')
log_file.flush()


def test(test_loader, n_share, C, sample_class_mean, precision, score_thr, classifier_type='classifier'):
    C.eval()
    per_class_num = np.zeros(n_share+1)
    per_class_correct = np.zeros(n_share+1).astype(np.float32)
    class_list = [i for i in range(n_share)]

    for batch_idx, (img, label) in enumerate(test_loader):
        print('num_iter: ', batch_idx)
        img, label = img.cuda(), label.cuda()
        score = lib_generation.get_Mahalanobis_score_batch(C, img, args.class_num, 'resnet', sample_class_mean, precision, 0.0005)
        C.eval()
        with torch.no_grad():
            if classifier_type == 'classifier':
                out, fea = C(img)
            elif classifier_type == 'Mahalanobis':
                out = score
            else:
                raise Exception('wrong classifier')
            if batch_idx == 0:
                open_class = int(out.size(1))
                class_list.append(open_class)
            pred = out.data.max(1)[1]
            print('pred: ', pred)
            ind_unk = np.where(score.max(1)[0].cpu() < score_thr)[0]
            print('max score: ', score.max(1)[0].cpu())
            print('ind_unk: ', ind_unk)
            pred[ind_unk] = open_class
            print('pred_: ', pred)
            print('label: ', label)
            pred = pred.cpu().numpy()
            for i, t in enumerate(class_list):
                t_ind = np.where(label.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
    print('per_class_correct:', per_class_correct)
    print('per_class_num:', per_class_num)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    print('h_score: '+str(h_score)+' known_acc: '+str(known_acc)+' Unknown_acc: '+str(unknown))
    return h_score, known_acc, unknown


def train():
    # set the datasets
    source_data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_data_transform = transforms.Compose([
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

    source_folder = ImageFolder(args.source_data, transform=source_data_transform)
    target_folder = ImageFolder(args.target_data, transform=target_data_transform)
    eval_folder = ImageFolder(args.target_data, transform=eval_data_transform)

    freq = Counter(source_folder.labels)
    # freq is a dict
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_folder.labels))
    source_loader = torch.utils.data.DataLoader(source_folder, batch_size=32, sampler=sampler, drop_last=True, num_workers=4)
    target_loader = torch.utils.data.DataLoader(target_folder, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(eval_folder, batch_size=32, shuffle=False, num_workers=4)

    source_estimator_loader = torch.utils.data.DataLoader(source_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    target_estimator_loader = torch.utils.data.DataLoader(target_folder, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    # set the models
    G = models.resnet.resnet50(pretrained=True)

    bottleneck = nn.Sequential(
        # nn.AdaptativeAvgPool2d(output_size=(1, 1)),
        # nn.Flatten(),
        nn.Linear(G.out_features, args.bottleneck_dim),
        nn.BatchNorm1d(args.bottleneck_dim),
        nn.ReLU()
    )

    C = models.basenet.Classifier(G, args.class_num, bottleneck, args.bottleneck_dim).to(torch.device('cuda'))

    C.load_state_dict(torch.load(args.pretrain_path))

    D = models.DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(torch.device('cuda'))

    # set the optimizer
    all_parameters = C.get_parameters() + D.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + 0.001 * float(x)) ** (-0.75))

    # set the loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        D, score_thr=args.score_thr, scale=args.scale,
        num_classes=args.class_num, features_dim=C._features_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim).to(torch.device('cuda'))

    # start training
    for epoch in range(args.num_epochs):
        log_file.write('epoch: '+str(epoch)+'\n')
        log_file.write('lr: '+str(lr_scheduler.get_last_lr())+'\n')
        log_file.flush()

        sample_class_mean, precision = lib_generation.sample_estimator(C, args.class_num, source_estimator_loader)

        # find the shared classes
        shared_class = list()
        Mahalanobis_t = lib_generation.get_Mahalanobis_score(C, target_estimator_loader, args.class_num, 'resnet', sample_class_mean, precision, 0.0005)
        num_per_class = dict()
        for i in range(args.class_num+1):
            num_per_class[i] = 0
        pred = Mahalanobis_t.max(1)[1]
        if epoch == 0:
            ind_unk = np.where(Mahalanobis_t.max(1)[0].cpu() < args.score_thr)[0]
        else:
            ind_unk = np.where(Mahalanobis_t.max(1)[0].cpu() < args.second_score_thr)[0]
        pred[ind_unk] = args.class_num
        pred = pred.cpu().numpy()
        for i in range(len(pred)):
            num_per_class[int(pred[i])] += 1
        max_num = 0
        for k in num_per_class:
            if k != args.class_num and num_per_class[k] > max_num:
                max_num = num_per_class[k]
        for k in num_per_class:
            if k != args.class_num and num_per_class[k] > max_num / 2.0:
                shared_class.append(k)
        log_file.write('shared_class: '+str(shared_class))
        log_file.flush()

        # calculate the h_score
        if epoch == 0:
            h_score, known_acc, Unknown_acc = test(test_loader, args.shared_class_num, C, sample_class_mean, precision, args.score_thr, 'classifier')
        else:
            h_score, known_acc, Unknown_acc = test(test_loader, args.shared_class_num, C, sample_class_mean, precision, args.second_score_thr, 'classifier')
        if epoch > 0:
            domain_adv.score_thr = args.second_score_thr
        log_file.write(' h_score :'+str(h_score)+' known_acc :'+str(known_acc)+' unknow_acc :'+str(Unknown_acc)+'\n')
        log_file.flush()

        # h_score, known_acc, Unknown_acc = test(test_loader, 10, C, sample_class_mean, precision, args.score_thr, classifier_type='Mahalanobis')
        # log_file.write(' h_score :'+str(h_score)+' known_acc :'+str(known_acc)+' unknow_acc :'+str(Unknown_acc)+'\n')
        # log_file.flush()

        C.train()
        domain_adv.train()

        data_iter_s = iter(source_loader)
        data_iter_t = iter(target_loader)
        len_train_source = len(source_loader)
        len_train_target = len(target_loader)

        total_cls_loss = 0.0
        total_transfer_loss = 0.0
        for i in range(args.iters_per_epoch):
            if i % len_train_source == 0:
                data_iter_s = iter(source_loader)
            if i % len_train_target == 0:
                data_iter_t = iter(target_loader)
            
            x_s, labels_s = next(data_iter_s)
            x_t, labels_t = next(data_iter_t)

            x_s = x_s.to(torch.device('cuda'))
            x_t = x_t.to(torch.device('cuda'))
            labels_s = labels_s.to(torch.device('cuda'))
            x = torch.cat((x_s, x_t), 0)

            # compute Mahalanobis score
            M = lib_generation.get_Mahalanobis_score_batch(C, x, args.class_num, 'resnet', sample_class_mean, precision, 0.0005)
            M_s, M_t = M.chunk(2, dim=0)

            # compute output
            y, f = C(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)
            if epoch ==0 and i < 10:
                log_file.write('iter: '+str(i)+'\n')
                log_file.write('the label of source batch: '+str(labels_s.data)+'\n')
                log_file.write('the cls result of Mahalanobis Detector: '+str(M_s.max(1)[1])+'\n')
                log_file.write('the cls result of Classifier: '+str(y_s.max(1)[1])+'\n')
                log_file.write('the label of target batch: '+str(labels_t.data)+'\n')
                log_file.write('the max score of target batch: '+str(M_t.max(1)[0])+'\n')
                log_file.write('the cls result of Mahalanobis Detector: '+str(M_t.max(1)[1])+'\n')
                log_file.write('the cls result of Classifier: '+str(y_t.max(1)[1])+'\n')

            cls_loss = F.cross_entropy(y_s, labels_s)
            weight_s = torch.ones(M_s.size()[0]).to(torch.device('cuda'))
            # ind_private = np.where(labels_s.cpu() in shared_class)[0]
            ind_private = list()
            for i, label in enumerate(labels_s.cpu().numpy()):
                if label not in shared_class:
                    ind_private.append(i)
            weight_s[np.array(ind_private)] = 0.0
            if epoch ==0 and i < 10:
                print('labels_s: ', labels_s)
                print('ind_private: ', ind_private)
            transfer_loss = domain_adv(M_s, f_s, M_t, f_t, weight_s)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * args.trade_off
            # loss = cls_loss

            total_cls_loss += cls_loss.data
            total_transfer_loss += transfer_loss.data

            # cls_acc = accuracy(y_s, labels_s)[0]

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        log_file.write(' cls_loss: '+str(total_cls_loss/args.iters_per_epoch)+' transfer_loss: '+str(total_transfer_loss/args.iters_per_epoch)+'\n')
        log_file.flush()
        # if epoch % 10 == 9:
        #     torch.save(C.state_dict(), os.path.join(output_path, 'epoch_{:05d}_model.pth.tar'.format(epoch)))
        torch.save(C.state_dict(), os.path.join(output_path, 'epoch_{:05d}_model.pth.tar'.format(epoch)))
    log_file.close()


def main():
    train()


if __name__ == '__main__':
    main()