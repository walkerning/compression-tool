# -*- coding: utf-8 -*-

import caffe
import numpy as np
import sys
import os


DEFAULT_MEAN_FILE_TYPE = 'npy'


def load_mean_file(file_name, file_type=None):
    file_type = file_type or os.path.splitext(file_name)[-1][1:] or DEFAULT_MEAN_FILE_TYPE
    return LOADERS[file_type](file_name)


def load_nparray_from_binary_proto(file_name):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(file_name, 'rb').read()
    blob.ParseFromString(data)
    return caffe.io.blobproto_to_array(blob)[0]


LOADERS = {
    'npy': np.load,
    'binaryproto': load_nparray_from_binary_proto
}
