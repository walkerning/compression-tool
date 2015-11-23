# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# package meta info
NAME = "svdtool"
VERSION=open("svdtool/VERSION").read().strip()
DESCRIPTION = "svdtool for fc layers of caffe model."
AUTHOR = ""
AUTHOR_EMAIL = ""

# package contents
MODULES = []
PACKAGES = find_packages()

ENTRY_POINTS = """
[console_scripts]
svd_tool=svdtool:svd_tool
"""

# dependencies
INSTALL_REQUIRES = ['protobuf', 'numpy']
TESTS_REQUIRE = []

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    py_modules=MODULES,
    packages=PACKAGES,

    package_data={'svdtool': ['VERSION']},

    entry_points=ENTRY_POINTS,

    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
