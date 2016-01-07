# -*- coding: utf-8 -*-

import logging
import yaml
import re
import numpy as np

from dan.base import BaseTool

def quantize_tool():
    raise Exception("Unimplemented")

class QuantizeTool(BaseTool):
    required_conf = ['input_proto', 'input_caffemodel', 'output_caffemodel', 'conditions']

    def __init__(self, config):
        super(QuantizeTool, self).__init__(config)

        self.quan_cond = config.conditions
        self.input_proto = str(config.input_proto)
        self.input_caffemodel = str(config.input_caffemodel)
        self.output_caffemodel = str(config.output_caffemodel)

    def run(self):
        logger = logging.getLogger('dan.quantool')
        import caffe
        import scipy.cluster.vq as scv

        net = caffe.Net(self.input_proto, self.input_caffemodel, caffe.TEST)
        layers = net.params.keys()
        done_layers = []

        for (regex, pattern, bits) in self.quan_cond:
            if regex:
                p = re.compile(pattern)
                def match(string):
                    res = p.match(string)
                    if res is None:
                        return False
                    span = res.span()
                    return (span[1] - span[0]) == len(string)
                
                layers_to_quan = filter(match, layers)
            else:
                layers_to_quan = filter(lambda x:pattern in x, layers)

            layers_to_quan = filter(lambda x:x not in done_layers, layers_to_quan)
            done_layers.extend(layers_to_quan)

            for layer in layers_to_quan:
                weights = net.params[layer][0].data

                W = weights.flatten()

                min_W = np.min(W)
                max_W = np.max(W)
                num_c = 2**bits - 1
                initial_uni = np.linspace(min_W, max_W, num_c)
                W_nonzero = W[W!=0]
                codebook, _ =  scv.kmeans(W_nonzero, initial_uni)
                while len(codebook) < num_c:
                    initial_uni = np.append(codebook, np.linspace(min_W, max_W, num_c))
                    codebook, _ =  scv.kmeans(W_nonzero, initial_uni)

                codebook = np.append(0, codebook)

                codes = scv.vq(W, codebook)
                quantized_data = codebook[codes]
                flatten_data = quantized_data.reshape(weights.shape)
                np.copyto(weights, flatten_data)
                logger.info('Finish quantization of layer %s!\n', layer)

        net.save(self.output_caffemodel)
        logger.info('Finish pruning of all layers! Caffemodel in file "%s".\n', self._log_output_caffemodel)
                
        return True
