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