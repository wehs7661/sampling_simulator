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
from sampling_simulator.wang_landau_algorithm import WL_Simulator


class EnsembleEXE(WL_Simulator):
    def __init__(self, params_dict, f_true):
        super().__init__(params_dict, f_true)
        self.params_dict = params_dict
        self.required_args.extend(['n_sim', 'n_iters', 's'])
        self.optional_args['w_combine'] = False
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
            print()
            print(section_title)
            print('=' * len(section_title))

            for j in range(self.n_sim):
                self.simulators[j].run()
                self.simulators[j].f_current = self.simulators[j].f_true - self.simulators[j].g

            # Update some attributes
            self.equil_all = [self.simulators[i].equil for i in range(self.n_sim)]
            self.equil_time_all = [self.simulators[i].equil_time if self.simulators[i].equil is True else None for i in range(self.n_sim)]  # noqa: E501
            if self.equil_all.count(True) == self.n_sim:
                print('The alchemical weights have been equilibrated in all replicas!')
                break

            self.wl_delta_all = [self.simulators[i].wl_delta for i in range(self.n_sim)]
            print(f'Final Wang-Landau incrementors: {np.round(self.wl_delta_all, decimals=6).tolist()}')

            if self.w_combine is True:
                print('Performing weight combination ...')
                weights_modified, g_vec = self.combine_weights()
                for j in range(self.n_sim):
                    self.simulators[j].g = weights_modified[j]

    def combine_weights(self):
        weights = [self.simulators[i].g for i in range(self.n_sim)]
        w = np.round(weights, decimals=3).tolist()  # just for printing
        print('  Original weights:')
        for i in range(self.n_sim):
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
        print('\n  Modified weights:')
        for i in range(len(w)):
            print(f'      States {i * self.s} to {i * self.s + self.n_sub - 1}: {w[i]}')

        return weights_modified, g_vec
