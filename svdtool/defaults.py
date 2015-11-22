# -*- coding: utf-8 -*-

ALL_METHOD = ['rank', 'accuracy_fall']

DEFAULT_METHOD = 'rank'

DEFAULT_METHOD_ARGUMENT = {
    'rank': '512',
    'accuracy_fall': '2'
}

METHOD_ARGUMENT_TRANSFORM = {
    'rank': int,
    'accuracy_fall': float
}
