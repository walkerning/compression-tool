# -*- coding: utf-8 -*-

import os
import logging

from dan.common.config import ConfigBunch

class BaseTool(object):
    """
    All tool class should be a subclass of this base tool"""
    
    # This list containis the names of required configurations of your tool
    required_conf = [] 

    def __init__(self, config):
        """
        Init your tool instance using a config bunch. (required!)

        Remember to call super(YourTool, self).__init__(config)
        """

        # hide file abs path, mainly for use of hiding file system details
        self.hide_file_path = getattr(config, 'hide_file_path', False)

    def __getattr__(self, name):
        """
        A magic method for protecting some information like the absolute file path. (in doubt)

        I think maybe this should be solved by chdir in the dan web and only log the \
        relative path.
        """
        # mainly for hide file path
        if name.startswith('_log_'):
            actual_name = name[5:]
            if actual_name in ['input_proto', 'output_proto', 'input_caffemodel',
                               'output_caffemodel']:
                if self.hide_file_path:
                    # fixme: maybe use platform-wise seperator
                    return os.path.join(*getattr(self, actual_name).rsplit('/', 2)[-2:])
                else:
                    return getattr(self, actual_name)
        # not all object suclass implement __getattr__
        super_obj = super(self.__class__, self)
        super_getattr = getattr(super_obj, '__getattr__', None)
        if super_getattr is not None:
            return super_getattr(name)
        else:
            raise AttributeError("type %s object has no attribute %s" % (self.__class__,
                                                                         name))
    @classmethod
    def populate_argument_parser(cls, parser):
        """
        Populate command line arguments for this tool. (optional)
        `parser` will be a argparse.ArgumentParser instance
        """

    @classmethod
    def load_from_config(cls, conf_dict, **kwargs):
        """
        Construct Tool instance from config dictionary.
        """
        logger = logging.getLogger('dan')
        for conf_name in cls.required_conf:
            if not conf_name in conf_dict:
                logger.error("In constructing %s: Configuration do not have '%s' which is required,"
                             " please check your configuration file.", cls.__name__,
                             conf_name)
                return None

        new_ins = cls(ConfigBunch(conf_dict), **kwargs)
        return new_ins

    def validate_conf(self):
        """
        Validate configuration after constructing, before running. (optional)

        * Return `True` when the configuration seems good to your tool.
        * Return `False` when the configuration is wrong.
        """
        return True

    def run(self):
        """
        Run the job! (required!)

        * Return `True` when success.
        * Raise Exception or return `False` when fail.
        
        If you want to log the path, use _log_input_proto or _log_input_caffemodel and so on."""

        return True
