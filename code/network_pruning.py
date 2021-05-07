import torch.nn.utils.prune as prune

import my_models2 as models # TREE stuff, need to adjust background-K
# import my_models as models # G2

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

from maxpoolPoolTest import get_args_parser, model_names

def main():

    ################################## 
    # Load the backbond model with checkpoint 
    # - first, init model architecture (e.g., ResNet) 
    ##################################   

    parser = get_args_parser()
    args = parser.parse_args()

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()


    model = torch.nn.DataParallel(model).cuda()

    ################################## 
    # Resume Checkpoint
    ##################################   
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


    ################################## 
    # Try Pruning Methods Here 
    # named_modules, named_parameters
    ##################################   
    parameter_filter_fun = lambda k, v : ((len(list(v.children())) == 0) and (k[:-1].endswith('conv') or k.endswith('fc')))

    parameter_to_prune = [
        (v, "weight") 
        for k, v in dict(model.named_modules()).items()
        if parameter_filter_fun(k, v)
    ]

    # now you can use global_unstructured pruning
    prune.global_unstructured(parameter_to_prune, pruning_method=prune.L1Unstructured, amount=0.8)

    # global sparsity
    nparams = 0
    pruned = 0
    for k, v in dict(model.named_modules()).items():
        if parameter_filter_fun(k, v):
            nparams += v.weight.nelement()
            pruned += torch.sum(v.weight == 0)
    print('Global sparsity across the pruned layers: {:.2f}%'.format( 100. * pruned / float(nparams)))

    # local sparsity
    for k, v in dict(model.named_modules()).items():
        if parameter_filter_fun(k, v):
            print(
                "Sparsity in {}: {:.2f}%".format(
                    k,
                    100. * float(torch.sum(v.weight == 0))
                    / float(v.weight.nelement())
                )
            )
    # ^^ will be different for each layer


    ################################## 
    # Before checkpointing, save the arguments. 
    ##################################   
    for k, v in dict(model.named_modules()).items():
        if parameter_filter_fun(k, v):
            prune.remove(v, 'weight')


    ################################## 
    # Now save the model into a checkpoint file 
    ##################################   

    torch.save({
            'epoch': 0, # just a dummy assignment 
            'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'best_acc1': best_acc1,
            # 'optimizer' : optimizer.state_dict(),
        },
        os.path.join( args.output_dir, 'pruning.pth.tar'),
        _use_new_zipfile_serialization=False 
        )

    return 


if __name__ == '__main__':
    main()