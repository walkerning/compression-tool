# -*- coding: utf-8 -*-

import logging
import re
import numpy as np

from dan.base import BaseTool
from dan.quantool import _quantize
from dan.prunetool import _prune

def nonmodel_tool():
    raise Exception("Unimplemented")

class blob(object):
    def __init__(self, W):
        self.data = W

class fakenet(object):
    '''
    A class that pretends to be caffe.Net
    '''
    def __init__(self, W):
        self.params = {}
        for key in W:
            if key[-2:] != '_b':
                self.params[key] = [blob(W[key])]
                self.params[key].append(blob(W[key + '_b']))

def _to_binary(array, types):
    bits = int(np.log2(types-1))+1
    if bits == 4:
        slots = 2
    elif bits == 8:
        slots = 1
    else:
        raise Exception("Not impemented %d-bit jump"%bits)
    stream_len =(len(array) -1)/slots+1
    stream = np.zeros(stream_len, np.uint8)
    for i in range(slots):
        data = array[np.arange(i, len(array), slots)]
        stream[:len(data)] += data * (2**(bits*i))

    return stream


def _stream_to_file(file_out, codebook, codes_W, net, ind_bits = 4, layers = None):
    if layers is None:
        layers = net.params.keys()
    fout = open(file_out,'wb')
    nz_num = np.zeros(len(layers), np.uint32)
    spm_stream = [0] * len(layers)
    ind_stream = [0] * len(layers)
    max_jump = 2 ** ind_bits

    for idx, layer in enumerate(layers):
        W = codes_W[layer].flatten()
        spm_tmp = np.zeros(W.size, dtype = np.uint32)
        ind_tmp = np.ones(W.size, dtype = np.uint32) * (max_jump-1)
        loc = np.where(W!=0)[0]
        distance_loc = np.append(loc[0], np.diff(loc)-1)  #jump 1 encode to 0
        zeros = distance_loc/max_jump
        idx_vec = np.cumsum(zeros+1)-1  #add the element itself. first one need -1
        total_slot = idx_vec[-1]+1
        nz_num[idx] = total_slot
        spm_tmp[idx_vec] = W[loc]
        ind_tmp[idx_vec] = distance_loc % max_jump

        spm_stream[idx] = _to_binary(spm_tmp[:total_slot], codebook[layer].size)
        ind_stream[idx] = _to_binary(ind_tmp[:total_slot], max_jump)

    nz_num.tofile(fout)
    for idx, layer in enumerate(layers):
        codebook[layer].astype(np.float32).tofile(fout)
        net.params[layer][1].data.tofile(fout)
        spm_stream[idx].tofile(fout)
        ind_stream[idx].tofile(fout)
    fout.close()

class PQTool(BaseTool):
    required_conf = ['input_npz', 'output_file', 'mode']

    TO_HIDE_PATH_ATTRS = ['output']

    def __init__(self, config):
        super(PQTool, self).__init__(config)

        self.weights = dict(np.load(config.input_npz))
        self.layers_rank = self.weights.pop('__rank__')
        self.output = config.output_file
        self.mode = config.mode
        self.validated = False

    def run(self):
        logger = logging.getLogger('dan.nonmodel_tool')
        if not self.validated:
            self.validate_conf()
        logger.info("================================")
        logger.info("Prune conditions")
        for condition in self.mode['prune_conditions']:
            logger.info("%10s %.2f"%(condition[1], condition[2]))
        logger.info("================================")
        logger.info("Quantize conditions")
        for condition in self.mode['quantize_conditions']:
            logger.info("%10s %2d"%(condition[1], condition[2]))
        logger.info("================================")

        net = fakenet(self.weights)
        _prune(net, self.mode['prune_conditions'], logger)
        codebook, codes_W = _quantize(net, self.mode['quantize_conditions'], logger)

        _stream_to_file(self.output, codebook, codes_W, net, ind_bits = 4, layers = self.layers_rank)

        logger.info('Finish all layers! Output bin file in "%s".\n', self._log_output)
                
        return True
    
    def validate_conf(self):
        self.validated = True
        if self.mode['foolmode']:
            self.mode['prune_conditions'] = []
            self.mode['quantize_conditions'] = []
            compress_rate = self.mode['compression_rate']

            if compress_rate < 4 or compress_rate > 20:
                return "Compression rate out of bound"

            layers = filter(lambda x: x[-2:] != '_b', self.weights.keys())
            unknown_layers = filter(lambda x:'conv' not in x and 'fc' not in x, layers)
            if len(unknown_layers) > 0:
                return "Unknown layers:" + unknown_layers[0]

            def rank_by_number(layers):
                get_number = lambda x:int(re.sub(r'[^[0-9]]*', '', x))
                rank = np.argsort(map(get_number, layers))
                return map(lambda x:layers[x], rank)

            conv_layers = filter(lambda x:'conv' in x, layers)
            fc_layers = filter(lambda x:'fc' in x, layers)
            conv_layers = rank_by_number(conv_layers)
            fc_layers= rank_by_number(fc_layers)

            params_number_c1 = self.weights[conv_layers[0]].size
            params_number_c2 = sum(map(lambda x:self.weights[x].size, conv_layers[1:]))
            params_number_f = sum(map(lambda x:self.weights[x].size, fc_layers))
            s = params_number_c1 + params_number_c2 + params_number_f
            p1 = float(params_number_c1) / s
            p2 = float(params_number_c2) / s
            q = float(params_number_f) / s

            nonzero_ratio = (8.0 / compress_rate - 3 * p1) / (3 * p2 + q)
            if nonzero_ratio < 0.25 :
                return "Compression rate unreachable"
            nonzero_ratio = min(nonzero_ratio, 1.0)

            self.mode['prune_conditions'].append([True, conv_layers[0], 0.0])
            self.mode['prune_conditions'].append([False, 'conv', 1-nonzero_ratio])
            self.mode['prune_conditions'].append([False, 'fc', 1 - nonzero_ratio / 2])
            self.mode['quantize_conditions'].append([False, 'conv', 8])
            self.mode['quantize_conditions'].append([False, 'fc', 4])
        return True

