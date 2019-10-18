# -*- coding: utf-8 -*-

import logging
import numpy as np

from google.protobuf import text_format

from dan.base import BaseTool
from dan.common.proto_utils import modify_message

def get_middle_layer_name(layer_name):
    return unicode(layer_name + '_mid')


def conv_tool():
    """
    Command line interface of the conv_tool."""

    pass


class ConvTool(BaseTool):
    required_conf = ['input_proto', 'input_caffemodel',
                     'output_proto', 'output_caffemodel',
                     'mode']

    TO_HIDE_PATH_ATTRS = ['input_proto', 'input_caffemodel',
                          'output_proto', 'output_caffemodel']

    def __init__(self, config):
        super(ConvTool, self).__init__(config)

        self.input_proto = str(config.input_proto)
        self.input_caffemodel = str(config.input_caffemodel)
        self.output_proto = str(config.output_proto)
        self.output_caffemodel = str(config.output_caffemodel)

        self.mode = config.mode
        self.validated = False

    def run(self):
        logger = logging.getLogger("dan.convtool")
        if not self.validated:
            validate_status = self.validate_conf()
            if not validate_status:
                return False

        import caffe

        new_solver = self._generate_new_proto(self.ori_solver, self.rank_spec)
                # 写入prototxt
        with open(self.output_proto, 'w') as output_proto_file:
            logger.info('Writing prototxt to "%s".', self._log_output_proto)
            output_proto_file.write(text_format.MessageToString(new_solver))

        # Construct the new net
        new_net = caffe.Net(self.output_proto, caffe.TEST)
        layer_index_dict = {name: i for i, name in enumerate(new_net._layer_names)}
        for layer_name, param in self.ori_net.params.iteritems():
            if layer_name not in self.rank_spec:
                update_blob_vec(new_net.layers[layer_index_dict[layer_name]].blobs, param)
            else:
                rank = self.rank_spec[layer_name]
                logger.info("Start decompositing convolution layer < %s >: retain %d kernels.",
                            layer_name, rank)
                u, v = self._decomposite_single_layer(param[0].data, rank)
                midlayer_name = get_middle_layer_name(layer_name)
                update_blob_vec(new_net.layers[layer_index_dict[layer_name]].blobs, (v,))
                # bias 设置在后一层
                update_blob_vec(new_net.layers[layer_index_dict[midlayer_name]].blobs, (u, param[1]))
                logger.info("Finish decompositing convolution layer < %s >: retain %d kernels.",
                            layer_name, rank)
        logger.info("Wring caffe model to file < %s >.", self._log_output_caffemodel)
        new_net.save(self.output_caffemodel)
        logger.info("Finish decompositing convolution layers! Prototxt in file < %s >"
                    ". Caffemodel in file < %s >.\n", self._log_output_proto,
                    self._log_output_caffemodel,
                    extra={
                        "important": True
                    })

        return True

    @classmethod
    def _generate_new_proto(cls, ori_solver, rank_spec):
        """Generate the new prototxt from the new rank specification of conv layers.

        Args:
            compression_spec (dict): the specification dict of how many times each layer should be compressed"""
        import caffe.proto.caffe_pb2 as caffepb2
        new_solver = caffepb2.NetParameter()
        new_solver.CopyFrom(ori_solver)
        new_solver.ClearField("layer")
        for i in range(len(ori_solver.layer)):
            layer = ori_solver.layer[i]
            if layer.name in rank_spec:
                mid_layer_name = get_middle_layer_name(layer.name)
                bottom_layer = modify_message(
                    layer,
                    in_place=False,
                    **{
                        'top': [mid_layer_name],
                        'convolution_param.num_output': rank_spec[layer.name],
                        'convolution_param.bias_term': False,
                        'param': [layer.param[0]]
                    }
                )
                mid_layer = modify_message(
                    layer,
                    in_place=False,
                    **{
                        'bottom': [mid_layer_name],
                        'name': mid_layer_name,
                        'convolution_param.stride': [1],
                        'convolution_param.kernel_size': [1],
                        'convolution_param.pad': [0]
                    }
                )
                new_solver.layer.extend([bottom_layer, mid_layer])
            else:
                new_solver.layer.append(layer)
        return new_solver

    @classmethod
    def _decomposite_single_layer(cls, data, rank):
        kernel_shape = data.shape[1:]
        data = data.reshape([data.shape[0], -1])
        u, s, v = np.linalg.svd(data, full_matrices=False)
        new_s = np.sqrt(s[0:rank])
        v = v[0:rank] * new_s[np.newaxis].T
        u = u[:, 0:rank] * new_s
        # reshape back to convolution size
        u = u[:, :, None, None]
        v = v.reshape([v.shape[0]] + list(kernel_shape))
        return u, v

    def validate_conf(self):
        logger = logging.getLogger("dan.convtool")

        if not self.mode.get("compression_specification", None):
            logger.error("Incompelete configuration: missing 'compression_specification'.")
            return False

        import caffe
        import caffe.proto.caffe_pb2 as caffepb2
        try:
            self.ori_net = caffe.Net(self.input_proto, *([self.input_caffemodel,
                                                 caffe.TEST] if
                                                self.input_caffemodel else [caffe.TEST]))
        except Exception as e:
            logger.error("Error occur when construct caffe.Net using %s and %s: %s", self._log_input_proto, self._log_input_caffemodel, e)
            return False

        # parse prototxt
        with open(self.input_proto, 'r') as input_proto_file:
            self.ori_solver = caffepb2.NetParameter()
            text_format.Merge(input_proto_file.read(), self.ori_solver)

        conv_layer_names = {l.name for l in self.ori_solver.layer if l.type == u'Convolution'}

        if not set(self.mode["compression_specification"]) < conv_layer_names:
            logger.error("Conv layers do not exist: < %s >. \t"
                         "Check your command line argument of '-l'. "
                         "Is that layer really a conv layer?",
                         ', '.join(set(self.mode["compression_specification"])
                                   - conv_layer_names))

        layer_index_dict = {layer.name: index for index, layer in enumerate(self.ori_solver.layer)}
        self.rank_spec = {}
        for layer_name, times in self.mode["compression_specification"].iteritems():
            try:
                times = int(times)
            except Exception as e:
                print e
            new_rank_number = self.ori_solver.layer[layer_index_dict[layer_name]].convolution_param.num_output / times
            self.rank_spec[layer_name] = new_rank_number

        self.validated = True
        return True


def update_blob_vec(old_blob_vec, new_data_vec):
    for i in range(len(new_data_vec)):
        new_data = new_data_vec[i]
        if not isinstance(new_data, np.ndarray) and hasattr(new_data, 'data'):
            new_data = new_data.data
        old_blob_vec[i].data[...] = new_data
