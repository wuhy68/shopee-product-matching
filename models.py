import sys
sys.path.append('')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from timm.models.layers import GroupNormAct, ClassifierHead, DropPath, AvgPool2dSame, create_pool2d, StdConv2d

import timm
from acblock.binaryneuraltree import *

''' I just wanted to understand and implement custom backward activation in PyTorch so I choose this.
    You can also simply use this function below too.

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return input * (torch.tanh(F.softplus(input)))
'''

class Mish_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]

        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f

class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Mish initialized")
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)

class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, n_dim, ls_=0.9):
        super().__init__()
        self.n_dim = n_dim
        self.ls_ = ls_

    def forward(self, x, target):
        target = F.one_hot(target, self.n_dim).float()
        target *= self.ls_
        target += (1 - self.ls_) / self.n_dim

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Model_Shopee(nn.Module):

    def __init__(self, model_name, out_dim):
        super(Model_Shopee, self).__init__()
        self.net = timm.create_model(model_name, pretrained=True)
        self.embedding_size = 512

        self.global_pool = GeM()

        if 'efficientnet' in model_name:
            self.net.classifier = nn.Identity()
            self.net.global_pool = nn.Identity()

        elif 'nfnet' in model_name:
            self.net.head.fc = nn.Identity()
            self.net.head.global_pool = nn.Identity()

        else:
            self.net.fc = nn.Identity()
            self.net.global_pool = nn.Identity()

        self.neck = nn.Sequential(
            nn.Linear(self.net.num_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.metric_classify = ArcMarginProduct_subcenter(self.embedding_size, out_dim)

    def extract(self, x):
        return self.net.forward_features(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.neck(x)

        logits_m = self.metric_classify(x)

        return F.normalize(x), logits_m

class Model_SD_Shopee(nn.Module):

    def __init__(self, model_name, out_dim):
        super(Model_SD_Shopee, self).__init__()
        self.net = timm.create_model(model_name, pretrained=True)
        self.net.reset_classifier(0, '')
        self.embedding_size = 512

        self.global_pool1 = GeM()
        self.global_pool = GeM()

        self.neck1 = nn.Sequential(
            nn.Linear(int(self.net.num_features / 2), self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.neck = nn.Sequential(
            nn.Linear(self.net.num_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.metric_classify = ArcMarginProduct_subcenter(self.embedding_size, out_dim)

    def extract(self, x):
        return self.net.forward_features(x, -1)

    def forward(self, x):
        x3, x = self.extract(x)

        x3 = self.global_pool1(x3)
        x3 = x3[:, :, 0, 0]
        x3 = self.neck1(x3)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.neck(x)

        logits_m3 = self.metric_classify(x3)
        logits_m = self.metric_classify(x)

        return F.normalize(x), logits_m3, logits_m




