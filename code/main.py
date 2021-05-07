import resnet_design3 as models # Design 3
# import resnet_design2 as models # Design 2
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import numpy as np
import logging
import pickle 
import pathlib

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch ImageNet Training', add_help=False)
    parser.add_argument('--data', metavar='DIR', default='/home/weixin/data/mix_mini_imagenet', 
                        help='path to dataset') # to the root of the meta-dataset 
    parser.add_argument('--task-num', default=5, type=int, metavar='N',
                        help='number of meta tasks used (default: 25, 4 classes in each task and 100 in total)')
    parser.add_argument('--background-K', default=1, type=int, metavar='N',
                        help='number of images total in the pool testing)')
    # parser.add_argument('--max-way', default=5, type=int, metavar='N',
    #                     help='number of way for the classifier (default: 5-way classification)') 
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
    # parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-valj', '--val-workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parser.add_argument('-p', '--print-freq', default=10, type=int,
    # parser.add_argument('-p', '--print-freq', default=500, type=int,
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
    # parser.add_argument('--seed', default=1234, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--log-name', default='ouptut.log', type=str, metavar='PATH',
                        help='path to the log file (default: output.log)')

    parser.add_argument('--output_dir', default='outputTmp',
                        help='path where to save, empty for no saving')

    return parser 

best_acc1 = 0


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)




    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    ##################################
    # Logging setting
    ##################################
    
    logging.basicConfig(
        filename=os.path.join( args.output_dir, args.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    print(str(args))
    logging.info( str(args) )

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)



    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    train_dataset_list = []
    val_dataset_list = []

    for folder_idx in range(args.task_num):
        traindir = os.path.join(args.data, str(folder_idx), 'train')
        valdir = os.path.join(args.data, str(folder_idx), 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print( "No. {}, traindir {}".format(folder_idx, traindir), "dataset len:", len(train_dataset.samples) )


        train_dataset_list.append(train_dataset)
        del train_dataset

        val_dataset = datasets.ImageFolder(
            valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        print( "No. {}, val_dataset {}".format(folder_idx, valdir), "dataset len:", len(val_dataset.samples) )
        val_dataset_list.append(val_dataset)
        del val_dataset

    ##################################
    # True group testing with fixed schedule 
    ##################################
    group_test_val_dataset = GroupTestDataset_val(val_dataset_list, args, split='val')
    group_test_val_loader = torch.utils.data.DataLoader(
        group_test_val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=True, 
        drop_last=False) 

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        back_bone_model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        back_bone_model = models.__dict__[args.arch]()

    ##################################
    # Modificaiton Happens in the backbone model 
    ##################################
    model = back_bone_model 
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            # raise NotImplementedError
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            raise NotImplementedError
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            ##################################
            # Current Going into This Branch
            ##################################
            model = torch.nn.DataParallel(model).cuda()



    # define loss function (criterion) and optimizer

    ##################################
    # Add Weights
    ##################################
    # weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
    # weights = weights / weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights).cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if args.evaluate:
        acc1 = validate(group_test_val_loader, model, criterion, args, dumpResult=True)
        return

    ##################################
    # No need to shuffle val data 
    ##################################
    # val_dataset = TaskCoalitionDataset_SuperImposing(val_dataset_list, args, split='val')
    # print( "len(val_dataset)", len(val_dataset) )
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.val_workers, pin_memory=True, 
    #     drop_last=False) 

    ##################################
    # Also construct a val single dataset 
    ##################################
    # val_dataset_K1 = TaskCoalitionDataset_SuperImposing(val_dataset_list, args, split='val', valK=0)
    # val_loader_K1 = torch.utils.data.DataLoader(
    #     val_dataset_K1,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.val_workers, pin_memory=True, 
    #     drop_last=False) 


    # if args.evaluate:
    #     # TODO: move it forward, and hack a dataset there. 
    #     acc1 = validate(val_loader, model, criterion, args, dumpResult=True)
    #     validate(val_loader_K1, model, criterion, args, dumpResult=False)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch, args)
        train(train_dataset_list, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args, dumpResult=True)
        ##################################
        # Also validate single image 
        ##################################
        # validate(val_loader_K1, model, criterion, args, dumpResult=False)

        ##################################
        # Test with fixed group testing schedule 
        ##################################
        acc1 = validate(group_test_val_loader, model, criterion, args, dumpResult=True)


        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args=args)


def train(train_dataset_list, model, criterion, optimizer, epoch, args):


    train_dataset = TaskCoalitionDataset_SuperImposing(train_dataset_list, args, split='train')
    print( "len(train_dataset)", len(train_dataset) )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.distributed: # added to support distributed 
            train_sampler.set_epoch(epoch)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)


    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        # [batch_time, losses, center_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))


    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        ##################################
        # multi-instance learning 
        ##################################
        # compute output
        output = model(images)
        loss = criterion(output, target) 


        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            progress.display(i)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def validate(val_loader, model, criterion, args, dumpResult=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        ##################################
        # Fields to be stored for postprocessing 
        ##################################
        target_all = []
        pred_score_all = [] 


        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)


            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1,5))
            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

            ##################################
            # For analysis
            ##################################
            output_scores = torch.nn.functional.softmax(output, dim=-1)
            positive_scores = output_scores[:,1]

            target_all.append( target.cpu().numpy() )
            pred_score_all.append( positive_scores.cpu().numpy() )

        target_all = np.concatenate( target_all, axis=0)
        pred_score_all = np.concatenate( pred_score_all, axis=0)

        
        if dumpResult is True:
            # with open(os.path.join( args.output_dir, 'model_validate_dump.pkl'), "wb") as pkl_file:
            with open(os.path.join( args.output_dir, 'model_validate_dump.pkl'), "wb") as pkl_file:
                pickle.dump( {
                    "target_all": target_all, 
                    "pred_score_all": pred_score_all, 
                    }, 
                    pkl_file 
                )
        
        # a large analysis here 
        pred_label = (pred_score_all>0.5)
        print("accuracy {:.3f}".format(accuracy_score(target_all, pred_label)) )
        print("roc_auc_score {:.3f}".format(roc_auc_score(target_all, pred_score_all)) )
        print("confusion_matrix\n{}".format(confusion_matrix(target_all, pred_label)))
        print("classification_report\n{}".format(classification_report(target_all, pred_label)))

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        print('VAL * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

        # if is_main_process():
        logging.info("accuracy {:.3f}".format(accuracy_score(target_all, pred_label)) )
        logging.info("roc_auc_score {:.3f}".format(roc_auc_score(target_all, pred_score_all)) )
        logging.info("confusion_matrix\n{}".format(confusion_matrix(target_all, pred_label)))
        logging.info("classification_report\n{}".format(classification_report(target_all, pred_label)))
        logging.info('VAL * Acc@1 {top1.avg:.3f}'
            .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None): 

    torch.save(state, os.path.join( args.output_dir, filename) )
    if is_best:
        shutil.copyfile(
            os.path.join( args.output_dir, filename), 
            os.path.join( args.output_dir, 'model_best.pth.tar')
            )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        ##################################
        # Save to logging 
        ##################################
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    # lr = args.lr * (0.1 ** (epoch // 5)) # doesn not work 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



##################################
# K - mixing 
# Implementation Assumption:
# - background tasks data num is at least twice as the positive data num 
##################################

class TaskCoalitionDataset_SuperImposing(torch.utils.data.Dataset):

    def __init__(self, dataset_list, args, split, valK=None):

        assert split in ['train', 'val'] 

        first_dataset = dataset_list[0] 
        self.loader = first_dataset.loader

        self.transform = first_dataset.transform 
        assert first_dataset.target_transform is None 

        self.classes = list()
        self.class_to_idx = dict()
        self.args = args 

        self.task_num = len(dataset_list)

        ##################################
        # Get all normal data first 
        ##################################
        positive_data_list = dataset_list[0].samples # weapon 
        normal_data_list = [ ] # normal classes in imagenet 
        for _, ds in enumerate(dataset_list[1:]):
            samples_this_ds = ds.samples
            normal_data_list.extend(samples_this_ds)

        normal_data_list = np.random.permutation(normal_data_list)

        ################################## 
        # Split for binary classification  
        ################################## 
        negative_data_list = normal_data_list[:len(positive_data_list)]   

        ################################## 
        # Redo Label Assignment and do mixing 
        ################################## 
        positive_target = 1 
        positive_data_list = [ [s[0], positive_target] for s in positive_data_list]
        negative_target = 0 
        negative_data_list = [ [s[0], negative_target] for s in negative_data_list]

        mixing_data_list = positive_data_list + negative_data_list 
        # print("mixing_data_list[0]", mixing_data_list[0])
        if split != 'val':
            mixing_data_list = np.random.permutation( mixing_data_list ) # preserve order 
        # print("mixing_data_list[0]", mixing_data_list[0])
        
        if split == 'train':
            # self.background_K = 1 #8-1 #(16-1) # 4 
            self.background_K = self.args.background_K
        elif split == 'val':
            if valK is None: 
                # self.background_K = 1 # 8-1 # (16-1) # 4 # 0: 96%, 1:96%, 4 - 94%, 9: 89%, 14:80.7% 19: 72% 
                self.background_K = self.args.background_K
            else:
                self.background_K = valK

        # self.background_K = 0 # 4 

        background_K_list = [ None for _ in range(self.background_K) ]
        for k_idx in range(self.background_K):
            k_idx_data_list = np.random.permutation(normal_data_list)[ :len(mixing_data_list)]
            background_K_list[k_idx] = k_idx_data_list

        ################################## 
        # Assign as members 
        # A list of lists
        # - First list: data mixing - postive or negative
        # - Second list - (K+1) list: background K lists 
        ################################## 
        self.dataset_samples = [mixing_data_list] + background_K_list

        ################################## 
        # Augment with normal training data 
        # Already in the train loop 
        ################################## 
        # if split == 'train':
        #     for folder_idx in range(self.background_K + 1): 
        #         # print("self.dataset_samples[folder_idx]", self.dataset_samples[folder_idx], type(self.dataset_samples[folder_idx]))
        #         self.dataset_samples[folder_idx] = np.concatenate( [self.dataset_samples[folder_idx], mixing_data_list], axis=0)
        #         # repeated. so that in same format  



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        ##################################
        # load T images
        ##################################

        superimposed_images = 0 
        mixing_folder_idx = 0
        target = int(self.dataset_samples[mixing_folder_idx][index][1]) # (path, target) 

        images_for_stack_list = []
        for folder_idx in range(self.background_K + 1): 
        # for folder_idx in [0]: 
            path, _ = self.dataset_samples[folder_idx][index]
            sample = self.loader(path)
            sample = self.transform(sample)
            images_for_stack_list.append(sample)

        # superimposed_images = superimposed_images / (self.background_K + 1) 
        superimposed_images = torch.stack(images_for_stack_list)
        
        return superimposed_images, target

    def __len__(self):
        return len(self.dataset_samples[0])

##################################
# K - mixing 
# Implementation Assumption:
# - background tasks data num is at least twice as the positive data num 
##################################

class GroupTestDataset_val(torch.utils.data.Dataset):

    def __init__(self, dataset_list, args, split, valK=None):

        assert split in ['val'] 

        first_dataset = dataset_list[0] 
        self.loader = first_dataset.loader

        self.transform = first_dataset.transform 
        assert first_dataset.target_transform is None 

        self.classes = list()
        self.class_to_idx = dict()
        self.args = args 

        self.task_num = len(dataset_list)

        ##################################
        # Get all normal data first 
        ##################################
        positive_data_list = dataset_list[0].samples # weapon 

        ################################## 
        # Redo Label Assignment and do mixing 
        # filter based on file name list 
        ################################## 
        import Constants
        positive_target = 1 

        ################################## 
        # Default Prevalence = 0.1%
        ################################## 
        positive_data_list = [ [s[0], positive_target] for s in positive_data_list if s[0].split('/')[-1] in Constants.firearm_file_paths]
        assert len(positive_data_list) == 50, len(positive_data_list)

        normal_data_list = [ ] # normal classes in imagenet 
        for _, ds in enumerate(dataset_list[1:]):
            samples_this_ds = ds.samples
            normal_data_list.extend(samples_this_ds)

        ################################## 
        # Split for binary classification  
        ################################## 
        negative_data_list = normal_data_list


        ################################## 
        # Overwrite to test on full test images (super noisy) 
        ################################## 
        # positive_data_list = [ [s[0], positive_target] for s in positive_data_list]
        # assert len(positive_data_list) == 150, len(positive_data_list)
        # negative_data_list = normal_data_list[:-100]

        ################################## 
        # Adjust Prevalence
        # Default: prevalence_percentage = 0.1%
        ################################## 

        # prevalence_percentage = 0.05 # 0.5 # 1.0 #

        prevalence_percentage = 0.1 # default 

        DEFAULT_prevalence_percentage = 0.1
        if prevalence_percentage == DEFAULT_prevalence_percentage:
            # no modification is needed 
            pass 

        elif prevalence_percentage > DEFAULT_prevalence_percentage:

            positive_data_list = positive_data_list * int(prevalence_percentage/DEFAULT_prevalence_percentage)
            num_negative_cutoff = len(positive_data_list) - 50
            negative_data_list = negative_data_list[num_negative_cutoff:]
            # OK, repeat positive. cut others 
            assert len(positive_data_list) == 250 # 500 # 
            pass
        elif prevalence_percentage == 0.01:
            # cut posivtive, repeat others 

            assert prevalence_percentage == 0.01
            positive_data_list = positive_data_list[::10] # half of the data 
            negative_data_list = negative_data_list + negative_data_list[:5] # extend 25 data points 
            assert len(positive_data_list) == 5

        elif prevalence_percentage == 0.05 :
            # cut posivtive, repeat others 

            assert prevalence_percentage == 0.05 
            positive_data_list = positive_data_list[::2] # half of the data 
            negative_data_list = negative_data_list + negative_data_list[:25] # extend 25 data points 

            assert len(positive_data_list) == 25  


        ################################## 
        # Redo Label Assignment and do mixing 
        ################################## 
        negative_target = 0 
        negative_data_list = [ [s[0], negative_target] for s in negative_data_list]

        # concat and shuffle 
        mixing_data_list = positive_data_list + negative_data_list 

        ################################## 
        # Consistent: Use seed 42 for all experiments. 
        # For non-adaptive testing that requires another seed: use 43 as the second seed. 
        ################################## 

        indices = torch.randperm( len(mixing_data_list), generator=torch.Generator().manual_seed(42)).tolist()
        # indices = torch.randperm( len(mixing_data_list), generator=torch.Generator().manual_seed(43)).tolist() # only for non-adaptive testing 

        shuffled_mixing_data_list = np.array(mixing_data_list)[indices]
        assert len(shuffled_mixing_data_list) == len(mixing_data_list) 

        ################################## 
        # Configure background K
        ##################################         
        if valK is None: 
            # self.background_K = 1 # 8-1 # (16-1) # 4 # 0: 96%, 1:96%, 4 - 94%, 9: 89%, 14:80.7% 19: 72% 
            self.background_K = self.args.background_K
        else:
            self.background_K = valK


        ################################## 
        # Reshape the list 
        ##################################   
        # print("shuffled_mixing_data_list", shuffled_mixing_data_list.shape)
        self.dataset_samples = np.array(shuffled_mixing_data_list).reshape( 
            len(shuffled_mixing_data_list)//(self.background_K + 1), 
            self.background_K + 1, 
            2
        ).transpose((1,0,2))


        # print("self.dataset_samples", self.dataset_samples.shape)      
        # print("self.dataset_samples[:,x]", self.dataset_samples[:,37])        
        # print("EXIT")
        # exit(0)
        # print("self.dataset_samples", self.dataset_samples)

        ################################## 
        # Dump the group testing schedule 
        ##################################   
        with open(os.path.join( args.output_dir, 'val_schedule.npy'), "wb") as npy_file:
            np.save(npy_file, self.dataset_samples)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        ##################################
        # load T images
        ##################################

        superimposed_images = 0 
        mixing_folder_idx = 0
        target = int(self.dataset_samples[mixing_folder_idx][index][1]) # (path, target) 

        images_for_stack_list = []
        for folder_idx in range(self.background_K + 1): 
        # for folder_idx in [0]: 
            path, target_this = self.dataset_samples[folder_idx][index]
            target = target or int(target_this) # Note: need to construct each target for group testing right now 
            sample = self.loader(path)
            sample = self.transform(sample)
            images_for_stack_list.append(sample)

        # superimposed_images = superimposed_images / (self.background_K + 1) 
        superimposed_images = torch.stack(images_for_stack_list)
        
        return superimposed_images, target

    def __len__(self):
        return len(self.dataset_samples[0])


if __name__ == '__main__':
    main()