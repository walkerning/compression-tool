#-*- coding: utf-8 -*-

# a very coarse test script just for save typing
import sys
sys.path.insert(0, sys.argv[1])

import logging
logging.basicConfig(level=logging.INFO)

import caffe
net = caffe.Net(sys.argv[2], sys.argv[3], caffe.TEST)

import dan
from dan.common import accuracy_utils
acc = accuracy_utils.get_network_acc_from_file(net, '/home/imgNet1k/imagenet1k_valid/',
                                               '/home/yaos11/nxf/tagList',
                                               test_num=5000,
                                               center_only=True,
                                               gpu=True, channel_swap='2,1,0',
                                               mean_file='/home/yaos11/yao/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')

print acc
