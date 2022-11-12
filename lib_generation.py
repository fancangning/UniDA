#***************************************************#
#ScriptName: lib_generation.py
#Author: fancangning@gmail.com
#Create Date: 2022-05-23 14:34
#Modify Authot: fancangning@gmail.com
#Modify Date: 2022-05-23 14:34
#Function: some functions about mahalanobis detector
#***************************************************#


from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform


def sample_estimator(model, num_classes, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean precision
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = list()
    for j in range(num_classes):
        list_features.append(0)
    
    with torch.no_grad():
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            output, out_features = model(data)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            
            # construct the sample matrix
            # print('out_features size:', out_features.size())
            # print('out_features[0] size:', out_features[0].size())
            # print('out_features[0] view size:', out_features[0].view(1, -1).size())
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    list_features[label] = out_features[i].view(1, -1)
                else:
                    list_features[label] = torch.cat((list_features[label], out_features[i].view(1, -1)), 0)
                num_sample_per_class[label] += 1
    
    sample_class_mean = torch.Tensor(num_classes, int(list_features[0].size()[1])).cuda()
    for j in range(num_classes):
        sample_class_mean[j] = torch.mean(list_features[j], 0)
    
    for i in range(num_classes):
        if i == 0:
            X = list_features[i] - sample_class_mean[i]
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean[i]), 0)
    
    group_lasso.fit(X.cpu().numpy())
    precision = group_lasso.precision_
    precision = torch.from_numpy(precision).float().cuda()

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score(model, test_loader, num_classes, net_type, sample_mean, precision, magnitude, return_label=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score
    '''
    model.eval()
    Mahalanobis = list()
    label = list()

    for index, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        data.requires_grad_()

        output, out_features = model(data)

        # compute Mahalanobis score
        # gaussian size batch_size * num_classes
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
        # input process
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean.index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        # print(type(data))
        # print(data.size())
        # print(type(data.grad))
        # print(data.grad.size())
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
        model.zero_grad()
        
        with torch.no_grad():
            output, noise_out_features = model(tempInputs)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
        
        if index == 0:
            Mahalanobis = noise_gaussian_score
        else:
            Mahalanobis = torch.cat((Mahalanobis, noise_gaussian_score), 0)

        if index == 0:
            label = target
        else:
            label = torch.cat((label, target), 0)
    if return_label:
        return Mahalanobis, label
    else:
        return Mahalanobis


def get_Mahalanobis_score_batch(model, data, num_classes, net_type, sample_mean, precision, magnitude):
    model.eval()
    data.requires_grad_()

    output, out_features = model(data)
    
    # compute Mahalanobis score
    # gaussian size batch_size * num_classes
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[i]
        zero_f = out_features - batch_sample_mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    
    # input process
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean.index_select(0, sample_pred)
    zero_f = out_features - batch_sample_mean
    pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    # print(type(data))
    # print(data.size())
    # print(type(data.grad))
    # print(data.grad.size())
    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    if net_type == 'resnet':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
    model.zero_grad()
    
    with torch.no_grad():
        output, noise_out_features = model(tempInputs)
    noise_gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[i]
        zero_f = noise_out_features.data - batch_sample_mean
        term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        if i == 0:
            noise_gaussian_score = term_gau.view(-1, 1)
        else:
            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

    data.requires_grad_(False)
    model.train()
    return noise_gaussian_score