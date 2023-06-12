import argparse
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AWA2', help='FLO, CUB')
    parser.add_argument('--root', default='', help='path to project')
    parser.add_argument('--image_root', default='', type=str, metavar='PATH',
                        help='path to image root')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=15, help='number of epochs to train for')
    parser.add_argument('--classifier_lr', type=float, default=1e-7, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=7048, help='manual seed 3483')
    parser.add_argument('--resnet_path', default='pretrained_models/resnet101-5d3b4d8f.pth',
                        # resnet101_cub.pth.tar resnet101-5d3b4d8f.pth
                        help="path to pretrain resnet classifier")
    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--image_type', default='test_unseen_loc', type=str, metavar='PATH',
                        help='image_type to visualize, usually test_unseen_small_loc, test_unseen_loc, test_seen_loc')
    parser.add_argument('--pretrain_epoch', type=int, default=1)
    parser.add_argument('--pretrain_lr', type=float, default=5e-5, help='learning rate to pretrain model')
    parser.add_argument('--all', action='store_true', default=True)
    parser.add_argument('--gzsl', action='store_true', default=True)
    parser.add_argument('--calibrated_stacking', type=float, default=2.0,
                        help='calibrated_stacking, shrinking the output score of seen classes')

    # for distributed loader
    parser.add_argument('--train_mode', type=str, default='random', help='loader: random or distributed')
    parser.add_argument('--n_batch', type=int, default=1000, help='batch numbers per epoch')
    parser.add_argument('--ways', type=int, default=16, help='class numbers per episode')
    parser.add_argument('--shots', type=int, default=2, help='image numbers per class')
    parser.add_argument('--train_whole_resnet', type=bool, default=True)
    parser.add_argument('--transform_complex', action='store_true', default=False, help='complex transform')
    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--resume', default=False)

    parser.add_argument('--att_size',type=int, default=85,help='size of the attribute')
    parser.add_argument('--ae_drop', type=int, default=0.2, help='drop rate')
    parser.add_argument('--res_size', type=int, default=2048, help='res_size')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--random_crop', type=bool, default=True)
    parser.add_argument('--train_beta', type=bool, default=True)
    parser.add_argument('--t', type=float, default=8)
    parser.add_argument('--Lp1', type=int, default=10)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.5)
    parser.add_argument('--seen_classes', type=int, default=40)
    parser.add_argument('--nclasses', type=int, default=50)
    parser.add_argument('--additional_loss', type=bool, default=True)
    parser.add_argument('--alpha1', type=float, default=5e-2)
    parser.add_argument('--alpha2', type=float, default=0.1)
    parser.add_argument('--alpha3', type=float, default=0.1)

    parser.add_argument('--model_path', default='',help="path to trained model.")

    # opt for finetune ALE
    opt = parser.parse_args()
    opt.dataroot = opt.root + 'data'
    opt.checkpointroot = opt.root + 'checkpoint'
    print('opt:', opt)
    return opt




