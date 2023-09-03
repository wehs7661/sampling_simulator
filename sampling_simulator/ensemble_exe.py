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
This module provide methods to mock the alchemical sampling in EEXE simulations.
"""
import numpy as np
from sampling_simulator.utils import utils
from sampling_simulator.wang_landau_algorithm import WL_Simulator


class EnsembleEXE(WL_Simulator):
    def __init__(self, params_dict, f_true):
        super().__init__(params_dict, f_true)
        self.params_dict = params_dict
        self.required_args.extend(['n_sim', 'n_iters', 's'])
        self.optional_args['w_combine'] = False
        self.optional_args['hist_correction'] = False
        self.check_params_dict()

        # Some EEXE-specific parameters
        self.n_sub = self.n_states - self.s * (self.n_sim - 1)
        start_idx = [i * self.s for i in range(self.n_sim)]
        self.state_ranges = [list(np.arange(i, i + self.n_sub)) for i in start_idx]
        self.equil_all = [None] * self.n_sim
        self.equil_time_all = [None] * self.n_sim

        # Initialize the simulators
        f_true_sub = [self.f_true[i * self.s:i * self.s + self.n_sub] for i in range(self.n_sim)]
        self.simulators = [WL_Simulator(self.params_dict, f_true_sub[i]) for i in range(self.n_sim)]

    def run(self):
        for i in range(self.n_iters):
            section_title = f'Iteration {i+1} / {self.n_iters}'
            if self.verbose:
                print()
                print(section_title)
                print('=' * len(section_title))
                print('Current alchemical weights:')
            for j in range(self.n_sim):
                self.simulators[j].run()
                self.simulators[j].f_current = self.simulators[j].f_true - self.simulators[j].g
                if self.verbose:
                    print(f'  States {j * self.s} to {j * self.s + self.n_sub - 1}: {np.round(self.simulators[j].g, decimals=3).tolist()}')  # noqa: E501

            # Update some attributes
            self.equil_all = [self.simulators[i].equil for i in range(self.n_sim)]
            self.equil_time_all = [self.simulators[k].equil_time + i * self.n_steps if self.simulators[k].equil is True else None for k in range(self.n_sim)]  # noqa: E501
            if self.equil_all.count(True) == self.n_sim:
                print('\nThe alchemical weights have been equilibrated in all replicas!')
                for j in range(self.n_sim):
                    print(f'  Equilibration time of states {j * self.s} to {j * self.s + self.n_sub - 1}: {self.equil_time_all[j]} steps')  # noqa: E501
                break

            self.wl_delta_all = [self.simulators[i].wl_delta for i in range(self.n_sim)]
            # if self.verbose:
            print(f'\nFinal Wang-Landau incrementors: {np.round(self.wl_delta_all, decimals=6).tolist()}')

            if self.w_combine is True:
                if self.verbose:
                    print('\nPerforming weight combination ...')
                weights_modified, g_vec = self.combine_weights()
                for j in range(self.n_sim):
                    self.simulators[j].g = weights_modified[j]
                    self.simulators[j].f_current = self.simulators[j].f_true - self.simulators[j].g
            else:
                # Calcualte g_vec but do not modify g and f_current
                _, g_vec = self.combine_weights()

            if self.verbose:
                print(f'Current profile of alchemical weieghts after combination: {np.round(g_vec, decimals=3).tolist()}')  # noqa: E501

        # Calculate RMSE for the whole-range alchemical weights
        self.rmse = utils.calc_rmse(g_vec, self.f_true)
        print(f'\nRMSE of the whole-range alchemical weights: {self.rmse:.3f} kT')

    def combine_weights(self):
        weights = [self.simulators[i].g for i in range(self.n_sim)]
        w = np.round(weights, decimals=3).tolist()  # just for printing
        if self.verbose:
            print('  Original weights:')
        for i in range(self.n_sim):
            if self.verbose:
                print(f'      States {i * self.s} to {i * self.s + self.n_sub - 1}: {w[i]}')

        dg_vec = []
        dg_adjacent = [list(np.diff(weights[i])) for i in range(self.n_sim)]
        for i in range(self.n_states - 1):
            dg_list = []
            for j in range(self.n_sim):
                if i in self.state_ranges[j] and i + 1 in self.state_ranges[j]:
                    idx = self.state_ranges[j].index(i)
                    dg_list.append(dg_adjacent[j][idx])
            dg_vec.append(np.mean(dg_list))
        dg_vec.insert(0, 0)
        g_vec = np.array([np.sum(dg_vec[:(i + 1)]) for i in range(len(dg_vec))])

        # Determine the vector of alchemical weights for each replica
        weights_modified = np.zeros_like(weights)
        for i in range(self.n_sim):
            if self.equil_all[i] is False:  # unequilibrated
                weights_modified[i] = list(g_vec[i * self.s: i * self.s + self.n_sub] - g_vec[i * self.s: i * self.s + self.n_sub][0])  # noqa: E501
            else:
                weights_modified[i] = self.simulators[i].g_equil

        w = np.round(weights_modified, decimals=3).tolist()  # just for printing
        if self.verbose:
            print('\n  Modified weights:')
        for i in range(len(w)):
            if self.verbose:
                print(f'      States {i * self.s} to {i * self.s + self.n_sub - 1}: {w[i]}')

        if self.hist_correction is True:
            # N' = N * exp(-(g' - g)), N' has to be an integer
            print('Performing histogram correction ...')
            for i in range(self.n_sim):
                # print('1: ', -(weights_modified[i] - weights[i]))
                print('2:, ', np.exp(-(weights_modified[i] - weights[i])))
                print(f'  Original histogram of states {i * self.s} to {i * self.s + self.n_sub - 1}: {self.simulators[i].hist.tolist()}')  # noqa: E501
                self.simulators[i].hist = (self.simulators[i].hist * np.exp(-(weights_modified[i] - weights[i]))).astype(int)  # noqa: E501
                print(f'  Corrected histogram of states {i * self.s} to {i * self.s + self.n_sub - 1}: {self.simulators[i].hist.tolist()}\n')  # noqa: E501

        return weights_modified, g_vec
