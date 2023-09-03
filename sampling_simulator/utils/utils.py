####################################################################
#                                                                  #
#    sampling_simulator,                                           #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2023 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
This modules provides utility functions for other modules.
"""
import copy
import numpy as np


def free2prob(f):
    """
    Convert a free energy profile to probabilities of all states.
    """
    f_ = copy.deepcopy(f)
    f_ -= f_.max()  # just to prevent overflow
    p = np.exp(-f_)
    p /= p.sum()

    return p


def calc_rmse(data, ref):
    """
    Calculate the root-mean-square error (RMSE) between the data and the reference.
    """
    return np.sqrt(((data - ref) ** 2).mean())


def get_subplot_dimension(n_panels):
    if int(np.sqrt(n_panels) + 0.5) ** 2 == n_panels:
        # perfect square number
        n_cols = int(np.sqrt(n_panels))
    else:
        n_cols = int(np.floor(np.sqrt(n_panels))) + 1

    if n_panels % n_cols == 0:
        n_rows = int(np.floor(n_panels / n_cols))
    else:
        n_rows = int(np.floor(n_panels / n_cols)) + 1

    return n_rows, n_cols
