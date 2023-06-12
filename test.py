from opt import get_opt
from DataSet import DATA_LOADER,get_loader
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np
import sys
import json
import os
import torch.nn as nn
from model import Encoder
from utils import Result,test_gzsl

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
class_attribute = data.attribute
result_gzsl = Result()
model = Encoder(opt)

if torch.cuda.is_available():
    model.cuda()
    class_attribute = class_attribute.cuda()

model.load_state_dict(torch.load(opt.model_path))

model.eval()
epoch=0
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