# -*- coding: utf-8 -*-

import os
import argparse
import logging

import sys

#import protobuf
import svdtool.defaults as defaults
from google.protobuf import text_format

from svdtool.utils import modify_message
from svdtool.common.utils import init_logging

file_dir = os.path.dirname(__file__)


def get_svd_layer_name(layer_name):
    return unicode(layer_name + '_svd_mid')


def verify_svd_method_module(mod):
    return getattr(mod, 'do_svd', None) is not None


def load_svd_method_module(method_name, mod_name=None):
    if mod_name is None:
        mod_name = "svd_method_" + method_name
    try:
        mod = __import__("svdtool." + mod_name, fromlist=["*"])
    except ImportError:
        raise

    if verify_svd_method_module(mod):
        return mod
    else:
        raise


def svd_tool():

    parser = argparse.ArgumentParser(
        description="SVD tool for fc layers of caffe network model."
    )

    parser.add_argument(
        '-l', '--layer', action='append'
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
        '-q', '--quiet', action='store_true'
    )
    parser.add_argument(
        '-c', '--caffe'
    )

    args = parser.parse_args()
    if args.caffe:
        sys.path.insert(0, args.caffe)

    init_logging(args.quiet)
    svd_tool_inner(args)

def svd_tool_inner(args):
    logger = logging.getLogger('svdtool')
    svd_spec_dict = dict(handle_input_arg(layer_string) for layer_string in args.layer)
    print svd_spec_dict
    import caffe
    import caffe.proto.caffe_pb2 as caffepb2
    net = caffe.Net(args.input_proto, *([args.input_caffemodel,
                                         caffe.TEST] if
                                        args.input_caffemodel else [caffe.TEST]))

    # parse prototxt
    input_proto_file = open(args.input_proto, 'r')
    solver = caffepb2.NetParameter()
    solver = text_format.Merge(input_proto_file.read(), solver)
    input_proto_file.close()

    new_solver = caffepb2.NetParameter()
    new_solver.CopyFrom(solver)
    new_solver.ClearField('layer')

    layer_index_dict = {}
    layer_index = 0
    # 构建第一个拆分的prototxt
    for i in range(len(solver.layer)):
        layer = solver.layer[i]
        if layer.name in svd_spec_dict:
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
        layer_index += 1

    # 写入拆分后的proto
    with open(args.output_proto, 'w') as output_proto_file:
        logger.info('Writing temporary prototxt file.')
        output_proto_file.write(text_format.MessageToString(new_solver))

    # 构建新的net方便计算
    new_net = caffe.Net(args.output_proto, caffe.TEST)

    final_solver = caffepb2.NetParameter()
    text_format.Merge(open(args.output_proto, 'r').read(), final_solver)

    final_param_dict = {}
    for layer_name, param in net.params.iteritems():
        if layer_name not in svd_spec_dict:
            continue
        svd_spec = svd_spec_dict[layer_name]
        hide_layer_size, new_param_list = svd_spec['method'](svd_spec['argument'],
                                                             param[0].data, net=net, new_net=new_net)

        svd_hidelayer_name = get_svd_layer_name(layer_name)

        # 存下来最后的数据
        final_param_dict[layer_name] = (new_param_list[0])
        modify_message(
            final_solver.layer[layer_index_dict[layer_name]],
            in_place=True,
            **{
                'inner_product_param.num_output': hide_layer_size
            }
        )
            #final_solver.layer[layer_index_dict[layer_name]].inner_product_param.num_output = hide_layer_size
        # bias设置在后一层
        final_param_dict[svd_hidelayer_name] = (new_param_list[1], param[1])


    # 读入新的prototxt，然后对需要赋值blobs的layer都赋值，最后save
    for layer_name, param in net.params.iteritems():
        if layer_name not in svd_spec_dict:
            # 其它层的layer.blobs就等于原来的blobs
            update_blob_vec(new_net.layers[layer_name].blobs,
                            param)
        else:
            svd_hidelayer_name = get_svd_layer_name(layer_name)
            update_blob_vec(new_net.layers[layer_index_dict[layer_name]].blobs,
                            final_param_dict[layer_name])
            update_blob_vec(new_net.layers[layer_index_dict[svd_hidelayer_name]].blobs,
                            final_param_dict[svd_hidelayer_name])


    logger.info('writing caffe model to file %s', args.output_caffe_model)
    new_net.save(args.output_caffe_model)
    logger.info('Finish processing svd. Prototxt in args.')


def update_blob_vec(old_blob_vec, new_data_vec):
    for i in range(len(new_data_vec)):
        new_data = new_data_vec[i]
        if hasattr(new_data, 'data'):
            new_data = new_data.data
        old_blob_vec[i].data[...] = new_data


def handle_input_arg(string):
    string = string.strip('[\t, ]+')
    args = string.split(',')
    svd_spec = dict(zip(['layer', 'method', 'argument'], args))
    svd_spec.setdefault('method', defaults.DEFAULT_METHOD)
    svd_spec.setdefault('argument', defaults.DEFAULT_METHOD_ARGUMENT[
        svd_spec['method']])
    svd_spec['argument'] = defaults.METHOD_ARGUMENT_TRANSFORM[svd_spec['method']](svd_spec['argument'])
    svd_spec['method'] = load_svd_method_module(svd_spec['method']).do_svd
    return args[0], svd_spec


if __name__ == "__main__":
    svd_tool()
