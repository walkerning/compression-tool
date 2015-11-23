# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

# package meta info
NAME = "dan"
VERSION=open("dan/VERSION").read().strip()
DESCRIPTION = "DAN: tools for compress caffe network."
AUTHOR = ""
AUTHOR_EMAIL = ""

# package contents
MODULES = []
PACKAGES = find_packages()

ENTRY_POINTS = """
[console_scripts]
svd_tool=dan:svd_tool
"""

# dependencies
INSTALL_REQUIRES = ['protobuf', 'numpy']
TESTS_REQUIRE = []

here = os.path.abspath(os.path.dirname(__file__))


def read_long_description(filename):
    path = os.path.join(here, 'filename')
    if os.path.exists(path):
        return open(path).read()
    return ''


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_long_description('README.md'),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    py_modules=MODULES,
    packages=PACKAGES,

    package_data={'dan': ['VERSION']},

    entry_points=ENTRY_POINTS,

    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
