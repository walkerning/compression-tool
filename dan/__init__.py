import os
from .svdtool import svd_tool

__version__ = open(os.path.join(os.path.dirname(__file__), "VERSION")).read().strip()