# -*- coding: utf-8 -*-

import yaml
import os

# http://stackoverflow.com/questions/528281/how-can-i-include-an-yaml-file-inside-another
class ConfigLoader(yaml.Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(ConfigLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, ConfigLoader)

ConfigLoader.add_constructor('!include', ConfigLoader.include)

class ConfigBunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
