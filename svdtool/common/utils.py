# -*- coding: utf-8 -*-
import logging
import os

from svdtool.common import log

def init_logging(quiet=False, colorize=True):
    level = logging.ERROR if quiet else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if colorize:
        stream_handler = log.ColorizingStreamHandler()
    else:
        stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def setup_glog_environ(quiet=False):
    if quiet:
        os.environ['GLOG_minloglevel'] = '2'
        # else遵从环境变量设置


def get_default_help(default, name):
    if isinstance(default, dict):
        return name + ': ' + '; '.join(['%s for %s'%(v, k) for k, v in default.iteritems()]) + '.\n'
    else:
        return name + ': ' + default + '.\n'
