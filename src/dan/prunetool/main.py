# -*- coding: utf-8 -*-

import logging
import yaml
import re
import numpy as np

from dan.base import BaseTool

def prune_tool():
    raise Exception("Unimplemented")

def _prune(net, prune_cond):
    layers = net.params.keys()
    done_layers = []
    
    for (regex, pattern, sparsity) in prune_cond:
        if regex:
            p = re.compile(pattern)
            def match(string):
                res = p.match(string)
                if res is None:
                    return False
                span = res.span()
                return (span[1] - span[0]) == len(string)
            
            layers_to_prune = filter(match, layers)
        else:
            layers_to_prune = filter(lambda x:pattern in x, layers)

        layers_to_prune = filter(lambda x:x not in done_layers, layers_to_prune)
        done_layers.extend(layers_to_prune)

        for layer in layers_to_prune:
            weights = net.params[layer][0].data

            flatten_data= weights.flatten()
            rank = np.argsort(abs(flatten_data))
            flatten_data[rank[:int(rank.size * sparsity)]] = 0

            flatten_data = flatten_data.reshape(weights.shape)
            np.copyto(weights, flatten_data)



class PruneTool(BaseTool):
    required_conf = ['input_proto', 'input_caffemodel', 'output_caffemodel', 'conditions']

    def __init__(self, config):
        super(PruneTool, self).__init__(config)

        self.prune_cond = config.conditions
        self.input_proto = str(config.input_proto)
        self.input_caffemodel = str(config.input_caffemodel)
        self.output_caffemodel = str(config.output_caffemodel)

    def run(self):
        logger = logging.getLogger('dan.prunetool')
        import caffe

        net = caffe.Net(self.input_proto, self.input_caffemodel, caffe.TEST)
        _prune(net, self.prune_cond)

        net.save(self.output_caffemodel)
        logger.info('Finish pruning of all layers! Caffemodel in file "%s".\n', self._log_output_caffemodel)
                
        return True

