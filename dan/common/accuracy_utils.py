# -*- coding: utf-8 -*-

import os
import glob
import caffe
import numpy as np
import pdb
import logging

import dataset_utils

logger = logging.getLogger('dan.common.accuracy_utils')

def get_inputs_from_file(input_file, ext, test_num):
    input_file = os.path.expanduser(input_file)
    if input_file.endswith('npy'):
        inputs = np.load(input_file)
        inputs = inputs[:min(test_num, len(inputs))]
    elif os.path.isdir(input_file):
        file_name_list = sorted(glob.glob(input_file + '/*.' + ext))
        inputs = []
        loaded_num = 0
        for file_name in file_name_list:
            inputs.append(caffe.io.load_image(file_name))
            loaded_num += 1
            if loaded_num >= test_num:
                break
    else:
        inputs = [caffe.io.load_image(input_file)]
    if len(inputs) < test_num:
        logger.warn('There is less than %d (test_num) pictures with extension %s '
                    'in the directory %s. Only %d inputs are loaded.',
                    test_num, ext, input_file, len(inputs))
    return inputs


def get_labels_from_file(label_file, test_num):
    label_file = os.path.expanduser(label_file)
    if label_file.endswith('npy'):
        labels = np.load(label_file)
    else:
        labels = [int(x) for x in open(label_file, 'r').read().strip().split('\n')]
    if len(labels) < test_num:
        logger.warn('There is less than %d (test_num) in the label file %s. '
                    'Only %d lables are loaded.', test_num,
                    label_file, len(labels))
        return labels
    else:
        return labels[:test_num]


def get_network_acc_from_file(net, input_file, label_file, test_num=1000,
                              mean_file=None, channel_swap=None,
                              images_dim="256,256", raw_scale=255,
                              input_scale=None, center_only=False,
                              ext=None, gpu=False, base_label=1):
    inputs = get_inputs_from_file(input_file, ext or 'JPEG', test_num)
    labels = get_labels_from_file(label_file, test_num)
    if len(inputs) != len(labels):
        logger.error('Inputs size %d not equal to label size %d. Cannot get network accuracy.',
                     len(inputs), len(labels))
        return None
    # mean and channel_swap
    mean, channel_swap = None, None
    if mean_file is not None:
        logger.warn('Mean file is deprecated. Ignore')
        #mean_file = os.path.expanduser(mean_file)
        #mean = dataset_utils.load_mean_file(mean_file)
        mean = np.array([123, 117, 104])#[104, 117, 123])
    if channel_swap is not None:
        channel_swap = [int(s) for s in channel_swap.split(',')]

    # image dimension
    image_dims = [int(s) for s in images_dim.split(',')]

    setup_net_for_classification(net,
                                 mean=mean,
                                 channel_swap=channel_swap,
                                 image_dims=image_dims,
                                 raw_scale=raw_scale,
                                 input_scale=input_scale,
                                 center_only=center_only,
                                 gpu=gpu)
    predictions = predict(net, inputs, not center_only)
    #np.save('try.npy', predictions)
    top5_prediction_labels = get_top5_prediction_labels(predictions)
    return get_accuracy(top5_prediction_labels, labels, base_label)


def setup_net_for_classification(net, mean=None, channel_swap=None,
                                 image_dims=None, raw_scale=None, input_scale=None,
                                 center_only=False, gpu=False):
    # gpu or cpu mode
    if gpu:
        caffe.set_mode_gpu()
        caffe.set_device(1)
    else:
        caffe.set_mode_cpu()

    in_ = net.inputs[0]

    # setup transformer
    from dan.common.caffe_utils import Transformer
    transformer = Transformer({in_: net.blobs[in_].data.shape})
    transformer.set_transpose(in_, (2, 0, 1)) # channel * H * W
    if mean is not None:
        transformer.set_mean(in_, mean)
    if input_scale is not None:
        transformer.set_input_scale(in_, input_scale)
    if raw_scale is not None:
        transformer.set_raw_scale(in_, float(raw_scale))
    if channel_swap is not None:
        transformer.set_channel_swap(in_, channel_swap)
    net.crop_dims = np.array(net.blobs[in_].data.shape[2:])
    if image_dims is None:
        image_dims = net.crop_dims
    net.image_dims = image_dims
    net.transformer = transformer


def get_top5_prediction_labels(predictions):
    top5_predictions = np.ndarray((len(predictions), 5), dtype=int)
    for index in range(len(predictions)):
        top5_predictions[index][...] = predictions[index].argsort()[::-1][:5]
    return top5_predictions


def get_accuracy(top5_prediction_labels, labels, base_label=0):
    top1_num = top5_num = 0
    label_num = len(labels)
    for index in range(label_num):
        label = labels[index] - base_label
        if label == top5_prediction_labels[index][0]:
            top1_num += 1
            top5_num += 1
        elif label in top5_prediction_labels[index]:
            top5_num += 1
    return float(top1_num) / label_num, float(top5_num) / label_num


def predict(self, inputs, oversample=True):
    """
    Predict classification probabilities of inputs.
    Parameters
    ----------
    inputs : iterable of (H x W x K) input ndarrays.
    oversample : boolean
    average predictions across center, corners, and mirrors
    when True (default). Center-only prediction when False.

    Returns
    -------
    predictions: (N x C) ndarray of class probabilities for N images and C
    classes.
    """
    # Scale to standardize input dimensions.
    input_ = np.zeros((len(inputs),
                       self.image_dims[0],
                       self.image_dims[1],
                       inputs[0].shape[2]),
                      dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, self.image_dims)

    if oversample:
        # Generate center, corner, and mirrored crops.
        input_ = caffe.io.oversample(input_, self.crop_dims)
    else:
        # Take center crop.
        center = np.array(self.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -self.crop_dims / 2.0,
            self.crop_dims / 2.0
        ])
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        #input_ = input_[:, 0:224, 0:224, :]

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
    import pdb
    pdb.set_trace()
    logger.info('starting forward all!')
    out = self.forward_all(**{self.inputs[0]: caffe_in})
    predictions = out[self.outputs[0]]

    # For oversampling, average predictions across crops.
    if oversample:
        predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        predictions = predictions.mean(1)

    return predictions
