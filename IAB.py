from opt import get_opt
from DataSet import DATA_LOADER,get_loader,map_label
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np
import sys
import json
import os
import torch.nn as nn
from model import Encoder,NAA
from utils import Result,test_gzsl,Loss_fn,SimMaxLoss,SimMinLoss
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''manual_seed'''
cudnn.benchmark = True
opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

'''get dataloader'''
data = DATA_LOADER(opt)
opt.test_seen_label = data.test_seen_label

# define test_classes
if opt.image_type == 'test_unseen_small_loc':
    test_loc = data.test_unseen_small_loc
    test_classes = data.unseenclasses
elif opt.image_type == 'test_unseen_loc':
    test_loc = data.test_unseen_loc
    test_classes = data.unseenclasses
elif opt.image_type == 'test_seen_loc':
    test_loc = data.test_seen_loc
    test_classes = data.seenclasses
else:
    try:
        sys.exit(0)
    except:
        print("choose the image_type in ImageFileList")

# Dataloader for train, test, visual
trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

# define attribute groups
if opt.dataset == 'CUB':
    parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
    group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8.json')))
    opt.resnet_path = 'pretrained_models/resnet101_c.pth.tar'
elif opt.dataset == 'AWA2':
    parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
             'character']
    group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
    opt.resnet_path = 'pretrained_models/resnet101-5d3b4d8f.pth'
elif opt.dataset == 'SUN':
    parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
    group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
    opt.resnet_path = 'pretrained_models/resnet101-5d3b4d8f.pth'
else:
    opt.resnet_path = 'pretrained_models/resnet101-5d3b4d8f.pth'
    parts = []
    group_dic = []

# prepare the attribute labels
class_attribute = data.attribute

print('Create Model...')
model = Encoder(opt)
NAA_model = NAA(opt,group_dic,data)
CAM_CRITERION = [SimMaxLoss(metric='cos',alpha=0.05).cuda(),SimMinLoss(metric='cos').cuda(),SimMaxLoss(metric='cos',alpha=0.05).cuda()]
'''cuda'''
if torch.cuda.is_available():
    model.cuda()
    NAA_model.cuda()
    class_attribute = class_attribute.cuda()

'''Save results'''
result_zsl = Result()
result_gzsl = Result()

print('Train and test...')
for epoch in range(opt.nepoch):
    # print("training")
    model.train()
    NAA_model.train()
    current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
    realtrain = epoch > opt.pretrain_epoch


    if epoch <= opt.pretrain_epoch:
        model.fix()
        '''optimizer'''
        model_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optim_params = [{'params': model_params}]
        if opt.train_beta:
            ie_parameters = [param for name, param in NAA_model.named_parameters()]
            optim_params.append({'params': ie_parameters,
                                 'lr': opt.pretrain_lr})
        optimizer = optim.Adam(optim_params, lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))

    else:
        model.nfix()
        print('start training resnet:')
        model_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optim_params = [{'params': model_params}]
        if opt.train_beta:
            ie_parameters = [param for name, param in NAA_model.named_parameters()]
            optim_params.append({'params': ie_parameters,
                                 'lr': opt.pretrain_lr})
        optimizer = optim.Adam(optim_params, lr=current_lr, betas=(opt.beta1, 0.999))

    batch = len(trainloader)

    neighbor_label = torch.zeros(opt.seen_classes*opt.Lp1,opt.seen_classes)
    for k in range(opt.seen_classes):
        neighbor_label[k*opt.Lp1:k*opt.Lp1+opt.Lp1,k] = 1
    neighbor_label = neighbor_label.cuda()
    to_average = []
    for i, (batch_input, batch_target, impath) in enumerate(trainloader):
        model.zero_grad()
        NAA_model.zero_grad()

        _ , attribute_seen , _ = NAA_model(class_attribute)
        attribute_seen = attribute_seen.cuda()
        batch_target = map_label(batch_target, data.seenclasses)
        input_v = Variable(batch_input)
        label_v = Variable(batch_target)
        if opt.cuda:
            input_v = input_v.cuda()
            label_v = label_v.cuda()
        top_k,v_logits,fg_feature,bg_feature = model(input_v,attribute_seen)
        loss = Loss_fn(opt,label_v,neighbor_label,top_k,CAM_CRITERION,fg_feature,bg_feature,v_logits,model.vars,realtrain)
        loss.backward()
        optimizer.step()
        to_average.append(loss.item() / opt.gamma)
    print('\n[Epoch %d]'% (epoch + 1),'Loss=',sum(to_average) / len(to_average))

    if (i + 1) == batch or (i + 1) % 200 == 0:
        model.eval()
        attribute_gzsl = class_attribute.T
        acc_GZSL_unseen,layer_acc_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
        acc_GZSL_seen,layer_acc_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)
        if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
            acc_GZSL_H = 0
        else:
            acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                    acc_GZSL_unseen + acc_GZSL_seen)
        if (layer_acc_unseen + layer_acc_seen) == 0:
            acc_layer_H = 0
        else:
            acc_layer_H = 2 * layer_acc_unseen * layer_acc_seen / (layer_acc_unseen + layer_acc_seen)

        if acc_GZSL_H > result_gzsl.best_acc:
            model_save_path = os.path.join('./out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
            torch.save(model.state_dict(), model_save_path)
            print('model saved to:', model_save_path)
        result_gzsl.update_gzsl(epoch + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H)
        print('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
              '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
              format(epoch + 1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U,
                     result_gzsl.best_acc_S,
                     result_gzsl.best_acc, result_gzsl.best_iter))



