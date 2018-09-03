"""Fitting code common to all optimal control models."""

import pickle
import copy
import os
import numpy as np
import simulation
from simulation import State, Risk, HIGH_LIVE_STATES, LOW_LIVE_STATES
import fit_risk_model
import fit_space_model

def randomise_init_infs(nodes, infect):
    """Randomise initial node state, matching given initial conditions.

    infect can be either tuple (HIGH, LOW) specifying numbers of high/low risk to infect randomly,
    or a dictionary {region: (HIGH, LOW)} to specify by region.
    """

    # First remove any current infection in nodes
    mynodes = copy.deepcopy(nodes)
    for node in mynodes:
        if node.state[State.INF_H] > 0:
            node.state[State.SUS_H] += node.state[State.INF_H]
            node.state[State.INF_H] = 0
        if node.state[State.INF_L] > 0:
            node.state[State.SUS_L] += node.state[State.INF_L]
            node.state[State.INF_L] = 0

    if isinstance(infect, tuple):
        nhigh = infect[0]
        nlow = infect[1]

        while nhigh + nlow > 0:
            node_choice = mynodes[np.random.choice(len(mynodes))]
            while node_choice.state[State.SUS_H] > 0 and nhigh > 0:
                node_choice.state[State.SUS_H] -= 1
                node_choice.state[State.INF_H] += 1
                nhigh -= 1
            while node_choice.state[State.SUS_L] > 0 and nlow > 0:
                node_choice.state[State.SUS_L] -= 1
                node_choice.state[State.INF_L] += 1
                nlow -= 1

    elif isinstance(infect, dict):
        regions = infect.keys()
        region_nodes = {region: [] for region in regions}
        for node in mynodes:
            region_nodes[node.region].append(node)

        for region in regions:
            nhigh = infect[region][0]
            nlow = infect[region][1]

            while nhigh + nlow > 0:
                node_choice = region_nodes[region][np.random.choice(len(region_nodes[region]))]
                while node_choice.state[State.SUS_H] > 0 and nhigh > 0:
                    node_choice.state[State.SUS_H] -= 1
                    node_choice.state[State.INF_H] += 1
                    nhigh -= 1
                while node_choice.state[State.SUS_L] > 0 and nlow > 0:
                    node_choice.state[State.SUS_L] -= 1
                    node_choice.state[State.INF_L] += 1
                    nlow -= 1

    else:
        raise TypeError("Wrong type for infect argument!")

    return mynodes

def randomise_infection(nodes, nhigh=0, nlow=0, nrand=0, single_loc=True, node_choice=None):
    """Initialise randomised infection in nodes, infecting inf_prop of susceptibles or ninf.

    If single_loc is True, then infecteds will be clumped in individual nodes, otherwise randomly
    positioned over landscape.
    """

    ninf = nhigh + nlow + nrand

    mynodes = copy.deepcopy(nodes)
    for node in mynodes:
        if node.state[State.INF_H] > 0:
            node.state[State.SUS_H] += node.state[State.INF_H]
            node.state[State.INF_H] = 0
        if node.state[State.INF_L] > 0:
            node.state[State.SUS_L] += node.state[State.INF_L]
            node.state[State.INF_L] = 0
    node_pops = [node.state[State.SUS_H] + node.state[State.SUS_L] for node in mynodes]

    if single_loc:
        while ninf > 0:
            if node_choice is None:
                node_choice = mynodes[np.random.choice(len(mynodes))]
            else:
                node_choice = mynodes[node_choice]
            while node_choice.state[State.SUS_H] > 0 and nhigh > 0:
                node_choice.state[State.SUS_H] -= 1
                node_choice.state[State.INF_H] += 1
                nhigh -= 1
                ninf -= 1
            while node_choice.state[State.SUS_L] > 0 and nlow > 0:
                node_choice.state[State.SUS_L] -= 1
                node_choice.state[State.INF_L] += 1
                nlow -= 1
                ninf -= 1
            while node_choice.state[State.SUS_H] + node_choice.state[State.SUS_L] > 0 and nrand > 0:
                probs = [node_choice.state[State.SUS_H], node_choice.state[State.SUS_L]]
                probs = probs / np.sum(probs)
                risk = np.random.choice([Risk.HIGH, Risk.LOW], p=probs)

                if risk == Risk.HIGH:
                    node_choice.state[State.SUS_H] -= 1
                    node_choice.state[State.INF_H] += 1
                else:
                    node_choice.state[State.SUS_L] -= 1
                    node_choice.state[State.INF_L] += 1

                nrand -= 1
                ninf -= 1
            node_choice = None

    else:
        host_nodes = np.concatenate([[i]*node_pop for i, node_pop in enumerate(node_pops)])
        np.random.shuffle(host_nodes)

        for i in range(ninf):
            probs = [mynodes[host_nodes[i]].state[State.SUS_H],
                     mynodes[host_nodes[i]].state[State.SUS_L]]
            probs = probs / np.sum(probs)
            risk = np.random.choice([Risk.HIGH, Risk.LOW], p=probs)

            if risk == Risk.HIGH:
                mynodes[host_nodes[i]].state[State.SUS_H] -= 1
                mynodes[host_nodes[i]].state[State.INF_H] += 1
            else:
                mynodes[host_nodes[i]].state[State.SUS_L] -= 1
                mynodes[host_nodes[i]].state[State.INF_L] += 1

    return mynodes

def logit_transform(params, bounds):
    """Logit transform parameters to remove bounds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_array = np.ma.array(
            [np.ma.log(np.true_divide((x - a), (b - x))) for x, (a, b) in zip(params, bounds)])
        ret_array.set_fill_value(0)
        return np.ma.filled(ret_array)

def reverse_logit_transform(params, bounds):
    """Reverse logit transform parameters to return bounds."""

    return np.array(
        [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for x, (a, b) in zip(params, bounds)])

class Fitter:
    """Class for fitting space and risk structured approximate models."""

    def __init__(self, nodes, risk_coupling, dist_coupling, sim_params):
        self.nodes = nodes
        self.risk_coupling = risk_coupling
        self.dist_coupling = dist_coupling
        self.sim_params = sim_params

        self.data = {}
        self.fits = {}

    def __repr__(self):
        repr_str = "\nOptimal Control Model Fitter\n" + "-"*20 + "\n\n"

        for fit_key in sorted(self.fits.keys()):
            repr_str += repr(self.fits[fit_key])

        return repr_str

    @classmethod
    def from_file(cls, filename="fitter.pickle"):
        """Initialise from pickle file."""

        with open(filename, "rb") as infile:
            fitter = pickle.load(infile)

        return fitter

    def save(self, filename="fitter.pickle"):
        """Pickle Fitter object to file."""

        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)

    def get_data(self, ninits=20, nrepeats=10, initial_nodes=None):
        """Run simulations and extract data for fitting.

        Arguments:
            ninits          Number of distinct initialisations (potentially changing initial conds.
            nrepeats        Number of simulation realisations for each initialisation.
            initial_nodes   Starting conditions for simulation realisations. If None then conditions
                            are randomised, otherwise must be a list of simulation.Node objects.
        """

        print("Generating training set...")

        self.data['dpc_times'] = np.linspace(0, self.sim_params['end_time'], 51)
        self.data['dpc_sus'] = np.empty((3, 2, len(self.data['dpc_times']), 0))
        self.data['dpc_inf'] = np.empty((3, 2, len(self.data['dpc_times']), 0))
        self.data['init_state'] = np.empty((0, 18))

        input_events = []

        for _ in range(ninits):
            if initial_nodes is None:
                nlow = np.random.randint(0, 3)
                nhigh = 2 - nlow
                nodes_init = randomise_infection(self.nodes, nhigh=nhigh, nlow=nlow)
            else:
                nodes_init = copy.deepcopy(initial_nodes)
            simulator = simulation.Simulator(
                nodes_init, self.risk_coupling, self.dist_coupling, self.sim_params)
            all_runs = simulator.run_simulation(nruns=nrepeats, verbose=False)
            if nrepeats == 1:
                all_runs = [all_runs]
            self._store_dpc_data(all_runs)

            input_events += self._extract_event_data(all_runs)

        self.data['input_events'] = input_events

    def fit_risk(self, bounds, start):
        """Fit risk model to generated data."""

        print("Fitting risk models...")

        if not {'input_events'} <= self.data.keys():
            raise RuntimeError("Must get data before fitting!")

        self.fits['risk'] = fit_risk_model.RiskFitter(self)
        self.fits['risk'].fit(bounds, start)

    def fit_space(self, bounds, start):
        """Fit space model to generated data."""

        print("Fitting space models...")

        if not {'input_events'} <= self.data.keys():
            raise RuntimeError("Must get data before fitting!")

        self.fits['space'] = fit_space_model.SpaceFitter(self)
        self.fits['space'].fit(bounds, start)

    def get_fit(self, space=False):
        """Get fitter given appropriate types."""

        if space:
            model_type = 'space'
        else:
            model_type = 'risk'

        fit_type = 'likelihood'

        fitter = self.fits[model_type].linear_fits[fit_type]

        return fitter

    def assess(self, save_folder=None, initial_nodes=None, max_control_rate=200, lik_map=False):
        """Assess fit quality for all model types and fit methods.

        Arguments:
            save_folder:        Where to save assessment plots.
            initial_nodes:      Node initialisation for simulations. If None then randomised,
                                otherwise list of simulation.Node objects.
            max_control_rate:   Maximum rate of vaccination for assessment under control.
            lik_map:            Whether to generate heat map of likelihood values.
        """

        model_types = self.fits.keys()

        for model_type in model_types:

            os.makedirs(os.path.join(save_folder, model_type), exist_ok=True)
            current_save_folder = os.path.join(save_folder, model_type)

            fitter = self.fits[model_type].linear_fits['likelihood']
            self.fits[model_type].optimise(
                fitter, initial_nodes=initial_nodes, max_control_rate=max_control_rate)

            # Linear assessments
            for method, fitter in self.fits[model_type].linear_fits.items():
                self.fits[model_type].assess(
                    fitter, save_folder=current_save_folder, control=None, likelihood_map=lik_map,
                    initial_nodes=initial_nodes)
                self.fits[model_type].assess(
                    fitter, save_folder=current_save_folder, max_control_rate=max_control_rate,
                    control=self.fits[model_type].data['opt_control'], initial_nodes=initial_nodes)

    def _store_dpc_data(self, all_runs, data_store=None, include_vac=False):
        """Append DPC data and initial states from a number of runs.

        Arguments:
            all_runs:       List of simulation.SimulationRun objects
            data_store:     Dictionary to save results to. If None save to self.data
            include_vac:    Whether to also store state of vaccinated population.
        """

        if data_store is None:
            data_store = self.data

        times = self.data['dpc_times']

        sus = np.zeros((3, 2, len(times), len(all_runs)))
        inf = np.zeros((3, 2, len(times), len(all_runs)))
        vac = np.zeros((3, 2, len(times), len(all_runs)))

        initial_state = np.zeros((1, 18))

        for run, (run_data, *_) in enumerate(all_runs):
            run_times0 = [x[0] for x in run_data["RegionA"]]
            run_times1 = [x[0] for x in run_data["RegionB"]]
            run_times2 = [x[0] for x in run_data["RegionC"]]
            for i, time in enumerate(times):
                idx0 = np.searchsorted(run_times0, time, side="right")
                idx1 = np.searchsorted(run_times1, time, side="right")
                idx2 = np.searchsorted(run_times2, time, side="right")
                states = [run_data["RegionA"][idx0-1][1:], run_data["RegionB"][idx1-1][1:],
                          run_data["RegionC"][idx2-1][1:]]

                for reg in range(3):
                    sus[reg, 0, i, run] = states[reg][State.SUS_H]
                    inf[reg, 0, i, run] = states[reg][State.INF_H]
                    vac[reg, 0, i, run] = states[reg][State.VAC_H]

                    sus[reg, 1, i, run] = states[reg][State.SUS_L]
                    inf[reg, 1, i, run] = states[reg][State.INF_L]
                    vac[reg, 1, i, run] = states[reg][State.VAC_L]

                if time == 0 and run == 0:
                    for reg in range(3):
                        for j, state_val in enumerate(sorted(HIGH_LIVE_STATES | LOW_LIVE_STATES)):
                            initial_state[0, 6*reg+j] = states[reg][state_val]

        names = ['dpc_sus', 'dpc_inf']
        data = [sus, inf]
        if include_vac:
            names += ['dpc_vac']
            data += [vac]

        for name, dat in zip(names, data):
            if name in data_store:
                data_store[name] = np.concatenate((data_store[name], dat), axis=3)
            else:
                data_store[name] = dat

        if 'init_state' in data_store:
            data_store['init_state'] = np.vstack((data_store['init_state'], initial_state))
        else:
            data_store['init_state'] = initial_state

    def _extract_event_data(self, all_runs):
        """Extract and store states and relevant event types for likelihood calculation.

        Returns:    List of numpy arrays, one for each run in all_runs.
                    Each array has columns:
                        0:12    State in each region before event
                                    S_A^H I_A^H S_A^L I_A^L S_B^H I_B^H etc
                        12      Time of event
                        13      1 if event is infection event, otherwise 0
                        14      Risk class of host about to be infected
                        15      Region of host event affects
        """

        ret_data = []

        for run in all_runs:
            run_data, event_data, *_ = run
            run_array = np.zeros((len(run_data['Global']), 16))

            previous_time = 0

            reg_iters = {
                "A": iter(np.array(run_data['RegionA'])),
                "B": iter(np.array(run_data['RegionB'])),
                "C": iter(np.array(run_data['RegionC']))
            }
            regs = {
                "A": next(reg_iters["A"]),
                "B": next(reg_iters["B"]),
                "C": next(reg_iters["C"])
            }

            for i, event_dat in enumerate(event_data):
                run_array[i, 0:4] = regs["A"][[1, 2, 5, 6]]
                run_array[i, 4:8] = regs["B"][[1, 2, 5, 6]]
                run_array[i, 8:12] = regs["C"][[1, 2, 5, 6]]

                time, event_id, _, new_state = event_dat
                region = self.nodes[event_id].region

                regs[region] = next(reg_iters[region])

                run_array[i, 12] = time - previous_time
                previous_time = time

                if new_state == State.INF_H or new_state == State.INF_L:
                    run_array[i, 13] = 1
                    if new_state == State.INF_H:
                        run_array[i, 14] = Risk.HIGH
                    elif new_state == State.INF_L:
                        run_array[i, 14] = Risk.LOW

                if region == "A":
                    run_array[i, 15] = 0
                elif region == "B":
                    run_array[i, 15] = 1
                elif region == "C":
                    run_array[i, 15] = 2

            run_array[-1, 0:4] = regs["A"][[1, 2, 5, 6]]
            run_array[-1, 4:8] = regs["B"][[1, 2, 5, 6]]
            run_array[-1, 8:12] = regs["C"][[1, 2, 5, 6]]
            run_array[-1, 12] = self.sim_params['end_time'] - previous_time
            run_array[-1, 13] = 0


            ret_data.append(run_array)

        return ret_data
