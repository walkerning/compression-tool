# -*- coding: utf-8 -*-

import argparse
import logging
import numpy
import re
import sys

from google.protobuf import text_format

from dan.base import BaseTool
from dan.svdtool import defaults
from dan.common.proto_utils import modify_message
from dan.common.utils import (init_logging, setup_glog_environ,
                              get_default_help)


def get_svd_layer_name(layer_name):
    return unicode(layer_name + '_svd_mid')


def verify_svd_method_module(mod):
    return getattr(mod, 'do_svd', None) is not None


def load_svd_method_module(method_name, mod_name=None):
    if mod_name is None:
        mod_name = "svd_method_" + method_name
    try:
        mod = __import__("dan.svdtool." + mod_name, fromlist=["*"])
    except ImportError:
        raise

    if verify_svd_method_module(mod):
        if not hasattr(mod.do_svd, 'method_name'):
            mod.do_svd.method_name = method_name
        if not hasattr(mod.do_svd, 'method_arg'):
            mod.do_svd.method_arg = method_name.upper()
        return mod
    else:
        raise Exception('SVD method module not legal!')


def svd_tool():
    """
    Command line interface of the svd-tool."""

    parser = argparse.ArgumentParser(
        description="SVD tool for fc layers of caffe network model."
    )

    SVDTool.populate_argument_parser(parser)

    args = parser.parse_args()
    if args.caffe:
        sys.path.insert(0, args.caffe)

    # init logging
    init_logging(args.quiet)
    setup_glog_environ(args.quiet or args.quiet_caffe)

    if hasattr(args, 'config_file') and args.config_file:
        svd_tool_ins = SVDTool.load_from_config_file(args.config_file)
    else:
        svd_tool_ins = SVDTool(args)

    if svd_tool_ins is not None:
        # run the inner svd tool
        status = svd_tool_ins.run()
    else:
        status = False

    sys.exit(0 if status else 1)

    
class SVDTool(BaseTool):
    required_conf = ['layers', 'input_proto', 'output_proto',
                     'input_caffemodel', 'output_caffemodel']

    def __init__(self, config):
        super(SVDTool, self).__init__(config)

        self.svd_spec_dict = dict(handle_input_arg(layer_string) for layer_string in config.layers)
        self.input_proto = str(config.input_proto)
        self.input_caffemodel = str(config.input_caffemodel)
        self.output_proto = str(config.output_proto)
        self.output_caffemodel = str(config.output_caffemodel)

        self.ori_solver = None
        self.ori_net = None

    @classmethod
    def populate_argument_parser(cls, parser):
        # Argument parser
        parser.add_argument(
            '-l', '--layers', action='append',
            metavar='NAME[,METHOD[,ARG]]',
            help=get_default_help(defaults.DEFAULT_METHOD, 'DEFAULT_METHOD') + \
            get_default_help(defaults.DEFAULT_METHOD_ARGUMENT, 'DEFAULT_ARG',),
            required=True
        )
        parser.add_argument(
            '--input-proto',
            required=True
        )
        parser.add_argument(
            '--input-caffemodel'
        )
        parser.add_argument(
            '--output-proto',
            required=True
        )
        parser.add_argument(
            '--output-caffemodel',
            required=True
        )

        parser.add_argument(
            '-q', '--quiet', action='store_true',
            help='Suppress all output whose critical level is less than WARN.'
        )

        parser.add_argument(
            '--quiet-caffe', action='store_true',
            help='Suppress caffe output whose critical level is less than WARN.'
        )
        parser.add_argument(
            '-c', '--caffe', help='The search path of pycaffe on your machine.'
        )

    def validate_conf(self):
        logger = logging.getLogger('dan.svdtool')
        import caffe
        import caffe.proto.caffe_pb2 as caffepb2
        try:
            net = caffe.Net(self.input_proto, *([self.input_caffemodel,
                                                 caffe.TEST] if
                                                self.input_caffemodel else [caffe.TEST]))
        except Exception as e:
            logger.error("Error occur when construct caffe.Net using %s and %s: %s", self._log_input_proto, self._log_input_caffemodel, e)
            return False

        if not set(self.svd_spec_dict) < set(net.params):
            if len(net.params) == 0:
                logger.error("Layers do not exist: < %s >. "
                             "Seems initialization of caffe.Net failed. Check the "
                             "caffe GLOG output for more details.",
                             ', '.join(set(self.svd_spec_dict) - set(net.params)))
            else:
                logger.error("Layers do not exist: < %s >. "
                             "Check your command line argument of '-l'. ",
                             ', '.join(set(self.svd_spec_dict) - set(net.params)))

            return False

        # parse prototxt
        with open(self.input_proto, 'r') as input_proto_file:
            solver = caffepb2.NetParameter()
            text_format.Merge(input_proto_file.read(), solver)

        if not solver.layer:
            logger.error("No layers could be loaded from the prototxt file < %s >. "
                         "Maybe you can use caffe/tools/upgrade_net_proto_text to upgdate "
                         "the prototxt file?",
                         self._log_input_proto)
            return False

        fc_layer_names = {l.name for l in solver.layer if l.type == u'InnerProduct'}

        if not set(self.svd_spec_dict).issubset(fc_layer_names):
            logger.error("FC layers do not exist: < %s >. \t"
                         "Check your command line argument of '-l'. "
                         "Is that layer really a fc layer?",
                         ', '.join(set(self.svd_spec_dict) - fc_layer_names))
            return False


        self.ori_solver = solver
        self.ori_net = net
        return True

    def run(self):
        logger = logging.getLogger('dan.svdtool')
        import caffe
        import caffe.proto.caffe_pb2 as caffepb2

        if self.ori_solver is None:
            validate_status = self.validate_conf()
            if not validate_status:
                return False

        new_solver = caffepb2.NetParameter()
        new_solver.CopyFrom(self.ori_solver)
        new_solver.ClearField('layer')

        layer_index_dict = {}
        layer_index = 0
        # 构建第一个拆分的prototxt
        for i in range(len(self.ori_solver.layer)):
            layer = self.ori_solver.layer[i]
            if layer.name in self.svd_spec_dict:
                mid_layer_name = get_svd_layer_name(layer.name)
                svd_bottom_layer = modify_message(
                    layer,
                    in_place=False,
                    **{
                        'top': [mid_layer_name]
                    }
                )
                svd_mid_layer = modify_message(
                    layer,
                    in_place=False,
                    **{
                        'bottom': [mid_layer_name],
                        'name': mid_layer_name
                    }
                )

                new_solver.layer.extend([svd_bottom_layer, svd_mid_layer])

                layer_index_dict[svd_bottom_layer.name] = layer_index
                layer_index += 1
                layer_index_dict[svd_mid_layer.name] = layer_index
            else:
                new_solver.layer.extend([layer])
                layer_index_dict[layer.name] = layer_index
            layer_index += 1

        # 写入拆分后的proto
        with open(self.output_proto, 'w') as output_proto_file:
            logger.info('Writing temporary prototxt to "%s".', self._log_output_proto)
            output_proto_file.write(text_format.MessageToString(new_solver))

        # 构建新的net方便计算
        new_net = caffe.Net(self.output_proto, caffe.TEST)

        final_solver = caffepb2.NetParameter()
        text_format.Merge(open(self.output_proto, 'r').read(), final_solver)

        final_param_dict = {}
        for layer_name, param in self.ori_net.params.iteritems():
            if layer_name not in self.svd_spec_dict:
                continue
            svd_spec = self.svd_spec_dict[layer_name]

            logger.info("Start calculating svd of layer < %s >. Strategy: %s. Argument < %s >: %s",
                        layer_name, svd_spec['method'].method_name,
                        svd_spec['method'].method_arg, str(svd_spec['argument']))
            hide_layer_size, new_param_list = svd_spec['method'](svd_spec['argument'],
                                                                 param[0].data, net=self.ori_net, new_net=new_net)
            logger.info("Finish calculating svd of layer < %s >.", layer_name)

            svd_hidelayer_name = get_svd_layer_name(layer_name)

            # Store the final data
            final_param_dict[layer_name] = (new_param_list[1],)
            modify_message(
                final_solver.layer[layer_index_dict[layer_name]],
                in_place=True,
                **{
                    'inner_product_param.num_output': hide_layer_size
                }
            )
            # bias设置在后一层
            final_param_dict[svd_hidelayer_name] = (new_param_list[0], param[1])

        with open(self.output_proto, 'w') as output_proto_file:
            logger.info('Writing proto to file "%s".', self._log_output_proto)
            output_proto_file.write(text_format.MessageToString(final_solver))

        new_net = caffe.Net(self.output_proto, caffe.TEST)
        # USE THIS, as caffe will insert some layer such as split
        # the `layer_index_dict` in the above code is of no use! TODO: remove those codes
        layer_index_dict = {name: i for i, name in enumerate(new_net._layer_names)}

        # 读入新的prototxt，然后对需要赋值blobs的layer都赋值，最后save
        for layer_name, param in self.ori_net.params.iteritems():
            if layer_name not in self.svd_spec_dict:
                # 其它层的layer.blobs就等于原来的blobs
                update_blob_vec(new_net.layers[layer_index_dict[layer_name]].blobs,
                                param)
            else:
                    svd_hidelayer_name = get_svd_layer_name(layer_name)
                    update_blob_vec(new_net.layers[layer_index_dict[layer_name]].blobs,
                                    final_param_dict[layer_name])
                    update_blob_vec(new_net.layers[layer_index_dict[svd_hidelayer_name]].blobs,
                                    final_param_dict[svd_hidelayer_name])

        logger.info('Writing caffe model to file "%s".', self._log_output_caffemodel)
        new_net.save(self.output_caffemodel)
        logger.info('Finish processing svd of fc layer! Prototxt in file "%s"'
                    '. Caffemodel in file "%s".\n', self._log_output_proto,
                    self._log_output_caffemodel,
                    extra={
                        'important': True
                    })

        return True


def update_blob_vec(old_blob_vec, new_data_vec):
    for i in range(len(new_data_vec)):
        new_data = new_data_vec[i]
        if not isinstance(new_data, numpy.ndarray) and hasattr(new_data, 'data'):
            new_data = new_data.data
        old_blob_vec[i].data[...] = new_data


def handle_input_arg(string):
    string = string.strip('[\t, ]+')
    args = re.split('[\t, ]+', string)
    svd_spec = dict(zip(['layer', 'method', 'argument'], args))
    svd_spec.setdefault('method', defaults.DEFAULT_METHOD)
    svd_spec.setdefault('argument', defaults.DEFAULT_METHOD_ARGUMENT[
        svd_spec['method']])
    svd_spec['argument'] = defaults.METHOD_ARGUMENT_TRANSFORM[svd_spec['method']](svd_spec['argument'])
    svd_spec['method'] = load_svd_method_module(svd_spec['method']).do_svd
    return args[0], svd_spec


if __name__ == "__main__":
    svd_tool()
