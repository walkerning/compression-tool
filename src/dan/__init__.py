# -*- coding: utf-8 -*-

import sys
import os
import argparse
import logging
import yaml

from setuptools import find_packages

from dan.base import BaseTool
from dan.common.utils import (init_logging, setup_glog_environ)
from dan.common.config import ConfigLoader
from dan.svdtool import svd_tool
from dan.__meta__ import (__version__, __author__, __title__,
                          __description__)

here = os.path.dirname(os.path.abspath(__file__))

def get_command_cls(package):
    name = getattr(package, "COMMAND_NAME", None)
    if name is None:
        return None
    class_obj = getattr(package, "COMMAND_CLASS", None)
    if class_obj is None:
        return None
    return (name, class_obj)

def load_command_packages():
    command_packages = find_packages(here, exclude=['common', 'common.*'])

    for package_name in command_packages:
        if '.' in package_name:
            # ignore inner packages
            continue
        try:
            # absolute importing
            package = __import__("dan." + package_name, fromlist=["*"])
        except ImportError:
            raise
            continue

        cls_obj_pair = get_command_cls(package)
        if cls_obj_pair:
            yield cls_obj_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--config-file',
        help='Configuration file of the dan util.',
        required=True)
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress all output whose critical level is less than WARN.')
    parser.add_argument(
        '--quiet-caffe', action='store_true',
        help='Suppress caffe output whose critical level is less than WARN.')
    parser.add_argument(
        '-c', '--caffe', help='The search path of pycaffe on your machine.'
    )

    args = parser.parse_args()

    init_logging(args.quiet)
    setup_glog_environ(args.quiet or args.quiet_caffe)

    if args.caffe:
        sys.path.insert(0, args.caffe)

    logger = logging.getLogger('dan')
    try:
        with open(args.config_file, 'r') as f:
            whole_config = yaml.load(f, ConfigLoader)
    except Exception as e:
        logger.error("ABORTING! %s: %s", e.__class__.__name__, e)
        sys.exit(1)

    if not 'pipeline' in whole_config or not 'config' in whole_config:
        logger.error("ABORTING! 'pipeline'/'config' entry must be in the configuration file! See the example_config.yaml for more details.")
        sys.exit(1)

    if whole_config.get('config', None) is None:
        whole_config['config'] = {}

    # 初始化所有stage的Command Worker
    worker_dict = {}
    command_cls_obj_dict = dict(load_command_packages())

    for stage in whole_config['pipeline']:
        stage_conf = whole_config['config'].get(stage, None)
        if stage_conf is None:
            logger.error("ABORTING! Configuration for stage '%s' not found, please check your configuration file.", stage)
            sys.exit(1)
        if 'command' not in stage_conf:
            logger.error("ABORTING! Configuration for stage '%s' do not have 'command' which is required, please check your configuration file.", stage)
            sys.exit(1)
        command_name = stage_conf.pop('command')
        cls_obj = command_cls_obj_dict.get(command_name, None)
        if cls_obj is None:
            logger.error("ABORTING! No such command exist: %s, please check your configuration file.", command_name)
            sys.exit(1)
        worker = cls_obj.load_from_config(stage_conf)
        if worker is None:
            logger.error("ABORTING! Due to configuration error of stage '%s'.", stage)
            sys.exit(1)
        if stage_conf.get('pre-validate', None):
            if not worker.validate_conf():
                logger.error("ABORTING! Pre-validate of configuration of stage '%s' fail.", stage)
                sys.exit(1)

        worker_dict[stage] = worker

    for stage in whole_config['pipeline']:
        try:
            status = worker_dict[stage].run()
        except Exception as e:
            # fixme: 暂时打印?
            print >>sys.stderr, e
            status = False
        if not status:
            logger.error("ABORTING! Error occur in stage '%s'", stage)
            break
        # explicitly delete
        # del worker_dict[stage] but there maybe several stage with the same name

    sys.exit(0 if status else 1)
