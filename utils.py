import torch
import numpy as np
from DataSet import map_label
import torch.nn as nn
import torch.nn.functional as F
class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.acc_list = []
        self.epoch_list = []
    def update(self, it, acc):
        self.acc_list += [acc]
        self.epoch_list += [it]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.epoch_list += [it]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s

def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = np.array(test_label)
    predicted_label = np.array(predicted_label)
    acc_per_class = []
    acc = np.sum(test_label == predicted_label) / len(test_label)
    for i in range(len(nclass)):
        idx = (test_label == i)
        acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
    return acc, sum(acc_per_class)/len(acc_per_class)


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
        acc_per_class = []
        acc = np.sum(test_label == predicted_label) / len(test_label)
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class.append(np.sum(test_label[idx] == predicted_label[idx]) / np.sum(idx))
        return acc, sum(acc_per_class)/len(acc_per_class)

def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)

def test_gzsl(opt, model, testloader, attribute, test_classes):
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    with torch.no_grad():
        for i, (input, target, impath) in \
                enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output = model(input, attribute)
            if opt.calibrated_stacking:
                output = calibrated_stacking(opt, output, opt.calibrated_stacking)
            _, predicted_label = torch.max(output.data, 1)

            predicted_label = predicted_label.int()
            predicted_layer = predicted_label.int()

            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())

            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc_gzsl(GT_targets,
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc_gzsl(GT_targets,
                                             np.array(predicted_layers), test_classes.numpy())
    if opt.all:
        return acc_all * 100,acc_layer_all*100
    else:
        return acc_avg * 100,acc_layer_avg*100


def cos_similarity(fg,bg):
    fg = F.normalize(fg,dim=1)
    bg = F.normalize(bg,dim=1)
    sim = torch.mm(fg,bg.T)
    return torch.clamp(sim,min=0.0005,max=0.9995)
class SimMinLoss(nn.Module):  #Minimize Similarity
    def __init__(self,margin = 0.15,metric = 'cos',reduction = 'mean'):
        super(SimMinLoss,self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction
    def forward(self,bg_feat,fg_feat):
        sim = cos_similarity(bg_feat,fg_feat)
        loss = -torch.log(1-sim)
        return torch.mean(loss)
class SimMaxLoss(nn.Module):
    def __init__(self,metric = 'cos',alpha = 0.25,reduction = 'mean'):
        super(SimMaxLoss,self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
    def forward(self,feat):
        sim = cos_similarity(feat,feat)
        loss = -torch.log(sim)
        loss[loss<0] = 0
        _,indices = sim.sort(dim = 1)
        _,rank = indices.sort(dim = 1)
        rank = rank-1
        rank_weights = torch.exp(-rank.float()*self.alpha)
        loss = loss*rank_weights
        return torch.mean(loss)

def label_adjust(opt,scores,label_v,neighbor_label):
    onehot_label = F.one_hot(label_v, opt.seen_classes)
    expanded_label = (onehot_label.unsqueeze(1) * neighbor_label.unsqueeze(0)).sum(-1)
    expanded_label = expanded_label.view(-1,opt.seen_classes,opt.Lp1)
    _, index = torch.topk(scores.view(-1, opt.seen_classes, opt.Lp1), dim=-1, k=opt.gamma)


    max_index = F.one_hot(index[:,:,0], opt.Lp1)+F.one_hot(index[:,:,1], opt.Lp1)
    expanded_label = (max_index*opt.delta+1)*expanded_label

    other_labels = torch.ones_like(expanded_label)
    other_labels = other_labels*(1-(opt.gamma*opt.delta/(opt.Lp1-opt.gamma)))
    expanded_label = torch.where(expanded_label==1,other_labels,expanded_label)
    return expanded_label.view(-1,opt.Lp1*opt.seen_classes)
def Loss_fn(opt,label_v,neighbor_label,top_k,CAM_CRITERION,fg_feats,bg_feats,vlogits,vars):
    loss = 0
    expanded_label = label_adjust(opt,top_k['cos'],label_v,neighbor_label)

    correct_in_top_k = (expanded_label * top_k['l2']).sum(-1)
    l_cls_l2 = -correct_in_top_k
    l_cls_l2 = l_cls_l2.mean()
    loss += l_cls_l2

    correct_in_top_k = (expanded_label * (top_k['cos'])).sum(-1)
    l_cls_cos = -correct_in_top_k
    l_cls_cos = l_cls_cos.neg().log().neg()
    l_cls_cos = l_cls_cos.mean()
    loss = loss + 0.05*l_cls_cos

    expanded_label_v = label_adjust(opt,vlogits,label_v,neighbor_label)
    l_reg = (expanded_label_v * vlogits).sum(-1)
    l_reg = -l_reg
    l_reg = l_reg.mean()
    loss = loss+0.1*l_reg

    loss1 = CAM_CRITERION[0](bg_feats)
    loss2 = CAM_CRITERION[1](bg_feats, fg_feats)
    loss3 = CAM_CRITERION[2](fg_feats)
    loss = loss + loss1 + loss2 + loss3

    if opt.additional_loss:
        correct_in_top_k = (expanded_label * top_k['cos_']).sum(-1)
        multy_loss = -correct_in_top_k
        l1 = multy_loss.mean()
        loss += 0.1 * l1

        weight_final = vars.activation_head.weight
        reg_loss = 5e-5 * weight_final.norm(2)
        loss += reg_loss
    return loss
