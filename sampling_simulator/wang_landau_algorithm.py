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
This module provide methods to mock the sampling in the alchemical space by the Wang-Landau algorithm.
The sampling in the configurational space is ignored, i.e., ΔU is assumed to be always 0.
kT is set to 1.
"""
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sampling_simulator.utils import utils


class WL_Simulator:
    def __init__(self, params_dict, f_true):
        for attr in params_dict:
            setattr(self, attr, params_dict[attr])
        self.n_states = len(f_true)
        self.f_true = copy.deepcopy(f_true)
        self.f_current = copy.deepcopy(f_true)  # unbiased upon initialization
        self.hist = np.zeros(self.n_states)
        self.g = np.zeros(self.n_states)
        self.state = 0  # starting from state 0
        self.traj = []  # state-space trajectory
        self.equil = False
        self.equil_time = None
        self.dg = []  # the weight difference between the first and last states
        self.required_args = [
            'n_steps',
            'wl_delta',
            'wl_delta_cutoff',
            'wl_ratio',
            'wl_scale',
        ]
        self.optional_args = {'verbose': False}
        self.check_params_dict()

    def check_params_dict(self):
        """
        Check if the required parameters are in the params_dict.
        """
        for arg in self.required_args:
            if not hasattr(self, arg):
                raise ParameterError(f'Required paramaeter {arg} is missing from the params_dict.')

        for arg in self.optional_args:
            if not hasattr(self, arg):
                setattr(self, arg, optional_args[arg])

    def check_flatness(self):
        """
        Check if the histogram is flat enough.
        """
        N_ratio = self.hist / self.hist.mean()
        flat_bool = np.all(N_ratio > self.wl_ratio) and np.all(1/N_ratio > self.wl_ratio)
        if flat_bool:
            if self.verbose:
                print('  Scaling down the Wang-Landau incrmentor and resetting the histogram ...')
            self.wl_delta *= self.wl_scale
            self.hist = np.zeros(self.n_states)
            if self.verbose:
                print(f'  New Wang-Landau incrmentor: {self.wl_delta:.6f}')

    def calc_prob_acc(self, state_new):
        """
        Calculate the acceptance probability of a proposed move.
        """
        delta = self.f_current[state_new] - self.f_current[self.state]
        if delta <= 0:
            p_acc = 1
        else:
            p_acc = np.exp(-delta)
        return p_acc

    def update(self, state_new):
        """
        For a given proposed state, calculates the accpetance probability,
        draw a random number to decide whether the propose move should be accpeted,
        and lastly, updates the histogram and weights.
        """
        p_acc = self.calc_prob_acc(state_new)
        rand = random.random()
        if rand < p_acc:
            if self.verbose:
                print('Move accepted!')
            self.state = state_new
            self.g[state_new] -= self.wl_delta
            self.f_current[state_new] += self.wl_delta
            self.hist[state_new] += 1
        else:
            if self.verbose:
                print('Move rejected!')
            self.g[self.state] -= self.wl_delta
            self.f_current[self.state] += self.wl_delta
            self.hist[self.state] += 1

        self.g -= self.g[0]

    def run(self):
        for i in range(self.n_steps):
            if self.verbose:
                print(f'Step {i + 1}: ', end='')
            self.traj.append(self.state)
            p_current = utils.free2prob(self.f_current)
            state_new = random.choices(range(self.n_states), weights=p_current, k=1)[0]
            if self.verbose:
                print(f'Attempting to move from state {self.state} to state {state_new} ... ', end='')
            self.update(state_new)
            self.dg.append(self.g[-1] - self.g[0])
            if not self.equil:
                self.check_flatness()
            if self.wl_delta < self.wl_delta_cutoff and self.equil is False:
                self.equil = True
                self.equil_time = i
                if self.verbose:
                    print('  The alchemical weights have been equilibrated!')

    def plot_hist(self):
        """
        Plot the histogram counts of all states.
        """
        plt.figure()
        bin_centers = np.arange(self.n_states)
        bin_width = bin_centers[1] - bin_centers[0]
        plt.bar(bin_centers, self.hist, width=bin_width, align='center', alpha=0.5, edgecolor='black')
        plt.xlabel('State index')
        plt.ylabel('Count')
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('hist.png', dpi=600)

    @staticmethod
    def plot_timeseries(var, label, fname):
        t = np.arange(len(var))
        plt.figure()
        if len(var) > 10000:
            plt.plot(t[::100], var[::100])
        else:
            plt.plot(t, var)
        plt.xlabel('Step')
        plt.ylabel(label)
        plt.grid()
        plt.savefig(fname, dpi=600)
