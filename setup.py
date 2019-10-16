# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

here = os.path.dirname(os.path.abspath((__file__)))
src_dir = os.path.join(here, 'src/')

# package meta info
meta_info = {}
with open(os.path.join(src_dir, "dan", "__meta__.py")) as f:
    exec(f.read(), meta_info)


# package contents
MODULES = []
PACKAGES = find_packages(where="src",
                         exclude=["tests", "tests.*"])

ENTRY_POINTS = """
[console_scripts]
dan=dan:main
svd_tool=dan:svd_tool
"""

# dependencies
# these upper/lower bound is just for the sake of my test environment
INSTALL_REQUIRES = ['protobuf',
                    # 'protobuf>=2.5.0',
                    # 'numpy>=1.9',
                    "numpy",
                    'pyyaml']
TESTS_REQUIRE = []

def read_long_description(filename):
    path = os.path.join(here, 'filename')
    if os.path.exists(path):
        return open(path).read()
    return ''


setup(
    name=meta_info['__title__'],
    version=meta_info['__version__'],
    description=meta_info['__description__'],
    long_description=read_long_description('README.md'),
    author=meta_info['__author__'],
    author_email=meta_info['__author_email__'],

    py_modules=MODULES,
    package_dir={"": "src"},
    packages=PACKAGES,

    #package_data={'dan': ['config_example.yaml']},
    include_package_data=True,

    entry_points=ENTRY_POINTS,
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
