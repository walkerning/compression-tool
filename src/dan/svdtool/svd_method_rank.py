# -*- coding: utf-8 -*-

import numpy as np

def do_svd(rank, params, *args, **kwargs):
    rank = min(params.shape + (rank,))
    u, s, v = np.linalg.svd(params, full_matrices=0)
    new_s = np.sqrt(s[0:rank])
    v = v[0:rank] * new_s[np.newaxis].T
    u = u[:, 0:rank] * new_s

    return rank, [u, v]

do_svd.method_name = "rank: Retain the biggest <RANK> singular value"
