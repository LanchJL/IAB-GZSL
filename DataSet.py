from sklearn import preprocessing
import sys
import torch.utils.data
import os
from PIL import Image
import numpy as np
import h5py
import torch
import torch.utils.data
import scipy.io as sio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label

'''read mat'''
class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        self.image_files = matcontent['image_files'].squeeze()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        # print("matcontent.keys:", matcontent.keys())
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        if opt.dataset == 'CUB':
            self.train_loc = matcontent['train_loc'].squeeze() - 1
            self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
            # self.train_unseen_loc = matcontent['train_unseen_loc'].squeeze() - 1

        # self.train_loc = matcontent['train_loc'].squeeze() - 1
        # self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[self.trainval_loc])
                _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[self.trainval_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[self.test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[self.test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
            self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[self.val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att


'''get label'''
def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label
'''open image'''
def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(opt, image_files, img_loc, image_labels, dataset):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if dataset == 'CUB':
            #print(image_file)
            image_file = opt.image_root+'/CUB/'+ image_file[0].split("MSc/")[1]
            #print(image_file,'########')
        elif dataset == 'AWA1':
            image_file = opt.image_root + image_file[0].split("databases/")[1]
        elif dataset == 'AWA2':
            image_file = opt.image_root + '/AwA2/JPEGImages/' + image_file[0].split("JPEGImages")[1]
        elif dataset == 'SUN':
            image_file = os.path.join(opt.image_root, image_file[0].split("data/")[1])
        elif dataset =='FLO':
            #print(image_file)
            image_file = opt.image_root+'FLO/jpg/'+image_file[0].split('/')[-1]

        else:
            exit(1)
        imlist.append((image_file, int(image_label)))
    return imlist


'''get image filelist'''
class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, opt, data_inf=None, transform=None, target_transform=None, dataset=None,
                 flist_reader=default_flist_reader, loader=default_loader, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if image_type == 'test_unseen_small_loc':
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'test_seen_loc':
            self.img_loc = data_inf.test_seen_loc
        elif image_type == 'trainval_loc':
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            self.img_loc = data_inf.train_loc
        else:
            try:
                sys.exit(0)
            except:
                print("choose the image_type in ImageFileList")


        if select_num != None:
            # select_num is the number of images that we want to use
            # shuffle the image loc and choose #select_num images
            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]
        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.dataset = dataset
        self.imlist = flist_reader(opt, self.image_files, self.img_loc, self.image_labels, self.dataset)
        self.image_labels = self.image_labels[self.img_loc]

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        #print(impath)
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        num = len(self.imlist)
        return num

class CategoriesSampler():
    # migrated from Liu et.al., which works well for CUB dataset
    def __init__(self, label_for_imgs, n_batch=1000, n_cls=16, n_per=3, ep_per_batch=1):
        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1
        # print('label_for_imgs:', label_for_imgs[:100])
        # print(np.unique(label_for_imgs))
        self.cat = list(np.unique(label_for_imgs))
        # print('self.cat', len(self.cat))
        # print(self.cat)
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)
        # print('self.catlocs[c]:', self.catlocs[0])

    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

'''get dataloader'''
def get_loader(opt, data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    Rsize = int(opt.image_size*8./7.)
    Isize = opt.image_size
    if opt.transform_complex:
        if opt.random_crop:
            train_transform = []
            train_transform.extend([
                transforms.Resize(Rsize),
                transforms.RandomCrop(Isize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize
            ])
            train_transform = transforms.Compose(train_transform)
            test_transform = []
            test_transform.extend([
                transforms.Resize(Isize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize
            ])
            test_transform = transforms.Compose(test_transform)
        else:
            train_transform = []
            train_transform.extend([
                transforms.Resize(Isize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize
            ])
            train_transform = transforms.Compose(train_transform)
            test_transform = []
            test_transform.extend([
                transforms.Resize(Isize),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize
            ])
            test_transform = transforms.Compose(test_transform)
    else:
        if opt.random_crop:
            train_transform = transforms.Compose([
                                          transforms.Resize(Rsize),
                                          transforms.CenterCrop(Isize),
                                          transforms.ToTensor(),
                                          normalize,
                                      ])
            test_transform = transforms.Compose([
                                                transforms.Resize(Rsize),
                                                transforms.CenterCrop(Isize),
                                                transforms.ToTensor(),
                                                normalize, ])
        else:
            train_transform = transforms.Compose([
                                          transforms.Resize(Isize),
                                          transforms.ToTensor(),
                                          normalize,
                                      ])
            test_transform = transforms.Compose([
                                                transforms.Resize(Isize),
                                                transforms.ToTensor(),
                                                normalize, ])

    dataset_train = ImageFilelist(opt, data_inf=data,
                                  transform=train_transform,
                                  dataset=opt.dataset,
                                  image_type='trainval_loc')
    if opt.train_mode == 'distributed':
        train_label = dataset_train.image_labels

        sampler = CategoriesSampler(
            train_label,
            n_batch=opt.n_batch,
            n_cls=opt.ways,
            n_per=opt.shots
        )
        trainloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_sampler=sampler, num_workers=4, pin_memory=True)
        # exit()
    elif opt.train_mode == 'random':
        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    dataset_test_unseen = ImageFilelist(opt, data_inf=data,
                                        transform=test_transform,
                                        dataset=opt.dataset,
                                        image_type='test_unseen_loc')
    testloader_unseen = torch.utils.data.DataLoader(
        dataset_test_unseen,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    if opt.random_crop:
        dataset_test_seen = ImageFilelist(opt, data_inf=data,
                                          transform=transforms.Compose([
                                              transforms.Resize(Rsize),
                                              transforms.CenterCrop(Isize),
                                              transforms.ToTensor(),
                                              normalize, ]),
                                          dataset=opt.dataset,
                                          image_type='test_seen_loc')
    else:
        dataset_test_seen = ImageFilelist(opt, data_inf=data,
                                          transform=transforms.Compose([
                                              transforms.Resize(Isize),
                                              transforms.ToTensor(),
                                              normalize, ]),
                                          dataset=opt.dataset,
                                          image_type='test_seen_loc')
    testloader_seen = torch.utils.data.DataLoader(
        dataset_test_seen,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # dataset for visualization (CenterCrop)
    dataset_visual = ImageFilelist(opt, data_inf=data,
                                   transform=transforms.Compose([
                                       transforms.Resize(Rsize),
                                       transforms.CenterCrop(Isize),
                                       transforms.ToTensor(),
                                       normalize, ]),
                                   dataset=opt.dataset,
                                   image_type=opt.image_type)

    visloader = torch.utils.data.DataLoader(
        dataset_visual,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return trainloader, testloader_unseen, testloader_seen, visloader