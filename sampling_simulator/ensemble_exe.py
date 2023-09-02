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
        self.check_params_dict()
        self.n_sub = self.n_states - self.s * (self.n_sim - 1)
        start_idx = [i * self.s for i in range(self.n_sim)]
        self.state_ranges = [list(np.arange(i, i + self.n_sub)) for i in start_idx]
        self.equil_all = [None] * self.n_sim
        self.equil_time_all = [None] * self.n_sim

    def run(self):
        f_true_sub = [self.f_true[i * self.s:i * self.s + self.n_sub] for i in range(self.n_sim)]
        self.simulators = [WL_Simulator(self.params_dict, f_true_sub[i]) for i in range(self.n_sim)]
        for i in range(self.n_iters):
            print(f'Iteration {i+1} / {self.n_iters}')
            for j in range(self.n_sim):
                print(f'Simulation for states {j * self.s} to {j * self.s + self.n_sub - 1}')
                self.simulators[j].run()
            self.equil_all = [self.simulators[i].equil for i in range(self.n_sim)]
            self.equil_time_all = [self.simulators[i].equil_time if self.simulators[i].equil is True else None for i in range(self.n_sim)]  # noqa: E501
            if self.equil_all.count(True) == self.n_sim:
                print('The alchemical weights have been equilibrated in all replicas!')
                break

            print('  Performing weight combination ...')
            weights_modified, g_vec = self.combine_weights()
            for j in range(self.n_sim):
                self.simulators[j].g = weights_modified[j]
                self.simulators[j].f_current = self.simulators[j].f_true + self.simulators[j].g

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


if __name__ == "__main__":
    params_dict = {
        'n_sim': 4,
        's': 7,
        'n_steps': 100,
        'n_iters': 100,
        'wl_ratio': 0.8,
        'wl_scale': 0.7,
        'wl_delta': 10,
        'wl_delta_cutoff': 0.001,
        'verbose': True
    }

    f_ref = np.array([
        0.0, 57.805215366068346, 112.49325354921746, 163.9053804543671, 211.1228680338599, 254.58059732433938, 295.5660812208014,
        334.6531501257331, 371.45476696195874, 406.65593569265764, 439.2891881651442, 469.42891547180665, 497.15580279170484,
        522.4463901739871, 545.4157353241036, 566.0988033555478, 584.5719734781715, 600.4976380135653, 614.3116862505102, 625.6622986339568,
        634.9000108527418, 639.2141393579959, 643.3695667349867, 647.2981063000773, 650.9040500232999, 652.5280053311945, 653.9774650452159,
        655.178901008137, 656.0135314031018, 656.2533557264542, 656.2030286485515, 655.7434099493724, 654.7837017140891, 653.4171450203427,
        651.8897886531751, 650.4268292093915, 649.155734577982, 647.8279229679827, 646.8756268046341, 645.4731180684097,
    ], dtype=float)

    simulator = EnsembleEXE(params_dict, f_ref)
    simulator.run()
