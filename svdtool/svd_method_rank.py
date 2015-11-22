# -*- coding: utf-8 -*-

import numpy as np

def do_svd(rank, params, *args, **kwargs):
    u, s, v = np.linalg.svd(params, full_matrices=0)
    new_s = np.squrt(s[0:rank])
    v = np.transpose(v[0:rank].transpose() * new_s)
    u = u[:, 0:rank] * new_s

    return rank, [u, v]
