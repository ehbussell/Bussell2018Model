"""
Spatially structured OCT model of network simulation, with optimisation methods.
"""

import os
import subprocess
import warnings
from collections import namedtuple
import argparse
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import bocop_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()


ModelRun = namedtuple("ModelRun", ["state", "costate", "control", "objective", "exit_code"])
ModelRun.__new__.__defaults__ = (None,)

def no_control_policy(time):
    """Policy carrying out no disease management."""

    return [0]*6

def even_control_policy(time):
    """Policy carrying out evenly distributed disease management."""

    return [0.16]*6

class IncFunc:
    """Infection incidence function."""

    def __init__(self, fitter):
        """Make incidence function."""

        self.predict = fitter.predict_rates
        self.sigma = fitter.data.get('sigma', np.ones((3, 3)))
        self.rho = fitter.data.get('rho', np.ones((1, 2)))

    def __call__(self, state):
        rates = self.predict(state[[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]])
        return rates[0, :]

class SpaceModel:
    """
    Class to implement space and risk structured OCT model.

    Initialisation requires the following dictionary of parameters:
        'birth_rate':       Birth rate,
        'death_rate':       Death rate,
        'removal_rate':     Removal rate,
        'recov_rate':       Recovery rate,
        'state_init':       Initial state,
        'times':            Times to solve for,
        'max_control_rate': Maximum vaccination rate,
        'high_alloc_cost':  Relative cost of allocation to high risk individuals
        'low_alloc_cost':   Relative cost of allocation to low risk individuals
    and a SpaceFitter object.
    """

    def __init__(self, params, space_fitter):
        self.required_keys = ['birth_rate', 'death_rate', 'removal_rate', 'recov_rate',
                              'state_init', 'times', 'max_control_rate',
                              'high_alloc_cost', 'low_alloc_cost']

        for key in self.required_keys:
            if key not in params:
                raise KeyError("Parameter {0} not found!".format(key))

        self.params = {k: params[k] for k in self.required_keys}

        for key in params:
            if key not in self.required_keys:
                warnings.warn("Unused parameter: {0}".format(key))

        self.inc_fn = IncFunc(space_fitter)
        self._mode = None
        self._controller = None

    def __repr__(self):
        ret_str = "<" + self.__class__.__name__ + "\n\n"
        for key in self.required_keys:
            ret_str += key + ": " + str(self.params[key]) + "\n"
        ret_str += ">"

        return ret_str

    def _print_msg(self, msg):
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))

    def state_deriv(self, time, state):
        """Return state derivative for model, including objective.

        Control should give *number* treated.
        """

        control = self._controller(time)
        host_states = state[:-1]

        inf_rates = self.inc_fn(np.array(state[:-1]))
        control_rates = [[
            self.params['max_control_rate'] * control[2*i+j] * host_states[6*i+3*j] /
            np.sum(host_states[(6*i+3*j):(3+6*i+3*j)])
            if np.sum(host_states[(6*i+3*j):(3+6*i+3*j)]) > 0 else 0
            for j in range(2)] for i in range(3)]

        d_sus = np.array([[
            self.params['birth_rate'] * np.sum(host_states[(6*i+3*j):(3+6*i+3*j)])
            - self.params['death_rate'] * host_states[6*i+3*j] - inf_rates[2*i+j]
            - control_rates[i][j] + self.params['recov_rate'] * host_states[1+6*i+3*j]
            for j in range(2)] for i in range(3)])

        d_inf = np.array([[
            inf_rates[2*i+j] - host_states[1+6*i+3*j] *
            (self.params['death_rate'] + self.params['removal_rate'] + self.params['recov_rate'])
            for j in range(2)] for i in range(3)])

        d_vac = np.array([[
            control_rates[i][j] - self.params['death_rate'] * host_states[2+6*i+3*j]
            for j in range(2)] for i in range(3)])

        d_obj = (
            np.sum(host_states[1::3]) + self.params['high_alloc_cost'] * np.sum(control[::2]) +
            self.params['low_alloc_cost'] * np.sum(control[1::2]))

        d_state = np.empty_like(state)
        d_state[0:-1:3] = d_sus.flatten()
        d_state[1:-1:3] = d_inf.flatten()
        d_state[2:-1:3] = d_vac.flatten()
        d_state[-1] = d_obj

        return d_state

    def run_policy(self, control_policy):
        """Run forward simulation using a given control policy."""

        self._mode = "POLICY"
        self._controller = control_policy

        initial_ode_value = np.zeros(19)
        initial_ode_value[:-1] = self.params['state_init']

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('vode', nsteps=10000, method='bdf')
        ode.set_initial_value(initial_ode_value, self.params['times'][0])

        xs = [self.params['state_init']]

        for time in self.params['times'][1:]:
            ode.integrate(time)
            xs.append(ode.y[:-1])

        state_t = interp1d(self.params['times'], np.vstack(xs).T, fill_value="extrapolate")
        control_t = interp1d(
            self.params['times'], np.array([control_policy(t) for t in self.params['times']]).T,
            fill_value="extrapolate")

        self._mode = None
        self._controller = None

        return ModelRun(state=state_t, costate=None, control=control_t, objective=ode.y[-1])

    def run_bocop(self, bocop_dir=None, verbose=True, cold_start=None, init_policy=None,
                  sol_file=None):
        """Run BOCOP solver and return optimal state, co-state and control.

        Returned control function gives number to treat as fn of time.
        """

        self._mode = "BOCOP"

        if init_policy is None:
            initialisation = self.run_policy(even_control_policy)
        else:
            initialisation = self.run_policy(init_policy)

        if bocop_dir is None:
            bocop_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SpaceModelBOCOP")

        set_bocop_params(self.params, self.inc_fn, cold_start=cold_start, init=initialisation,
                         sol_file=sol_file, folder=bocop_dir)

        if verbose is True:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")], cwd=bocop_dir)
        else:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")],
                           cwd=bocop_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        if sol_file is None:
            sol_file = "problem.sol"

        state_t, costate_t, control_t, exit_text = bocop_utils.read_sol_file(
            os.path.join(bocop_dir, sol_file), ignore_fail=True)

        self._mode = None

        objective_value = state_t(self.params['times'][-1])[-1]

        return ModelRun(state=state_t, costate=costate_t, control=control_t,
                        objective=objective_value, exit_code=exit_text)

def set_bocop_params(params, inc_fn, cold_start=None, init=None, sol_file=None,
                     folder="SpaceModelBOCOP"):
    """Save parameters and initial conditions to file for BOCOP optimisation.

    Must provide either a .sol file for cold start, or a ModelRun object to initialise from (cold
    start takes priority). Solution file will be saved to sol_file location.
    """

    if sol_file is None:
        sol_file = "problem.sol"

    with open(os.path.join(folder, "problem.bounds"), "r") as infile:
        all_lines = infile.readlines()

    # Initial conditions
    for i in range(18):
        all_lines[9+i] = str(params['state_init'][i])+" "+str(params['state_init'][i])+" equal\n"
    all_lines[9+18] = "0 0 equal\n"

    # Max budget expenditure
    all_lines[63] = "-2e+020 1.0 upper\n"

    with _try_file_open(os.path.join(folder, "problem.bounds")) as outfile:
        outfile.writelines(all_lines)

    # Constant values
    with open(os.path.join(folder, "problem.constants"), "r") as infile:
        all_lines = infile.readlines()

    all_lines[5] = str(params['birth_rate']) + "\n"
    all_lines[6] = str(params['death_rate']) + "\n"
    all_lines[7] = str(params['removal_rate']) + "\n"
    all_lines[8] = str(params['recov_rate']) + "\n"
    all_lines[9] = str(params['max_control_rate']) + "\n"
    for i in range(9):
        all_lines[10+i] = str(inc_fn.sigma.flatten()[i]) + "\n"
    for i in range(4):
        all_lines[19+i] = str(inc_fn.rho.flatten()[i]) + "\n"
    all_lines[23] = str(params['high_alloc_cost']) + "\n"
    all_lines[24] = str(params['low_alloc_cost']) + "\n"

    with _try_file_open(os.path.join(folder, "problem.constants")) as outfile:
        outfile.writelines(all_lines)

    with open(os.path.join(folder, "problem.def"), "r") as infile:
        all_lines = infile.readlines()

    n_steps = str(len(params['times']) - 1)
    all_lines[5] = "time.initial double " + str(params['times'][0]) + "\n"
    all_lines[6] = "time.final double " + str(params['times'][-1]) + "\n"
    all_lines[18] = "discretization.steps integer " + n_steps + "\n"
    all_lines[107] = "solution.file string " + sol_file + "\n"

    # Initialisation
    if cold_start is not None:
        # Initialise from solution file
        all_lines[31] = "initialization.type string from_sol_file_cold\n"
        all_lines[32] = "initialization.file string " + cold_start + "\n"
        with _try_file_open(os.path.join(folder, "problem.def")) as outfile:
            outfile.writelines(all_lines)
    elif init is not None:
        # Initialise from ModelRun
        all_lines[31] = "initialization.type string from_init_file\n"
        all_lines[32] = "initialization.file string none\n"
        with _try_file_open(os.path.join(folder, "problem.def")) as outfile:
            outfile.writelines(all_lines)

        control_init = np.array([[init.control(t)[j] for j in range(6)] for t in params['times']])

        for control in range(6):
            all_lines = [
                "#Starting point file\n",
                "# This file contains the values of the initial points\n",
                "# for variable control #{0}\n".format(control), "\n",
                "# Type of initialization :\n", "linear\n", "\n",
                "# Number of interpolation points :\n", "{0}\n".format(len(params['times'])), "\n",
                "# Interpolation points :\n"]

            for i, time in enumerate(params['times']):
                all_lines.append("{0} {1}\n".format(time,
                                                    np.round(control_init[i, control], decimals=2)))

            with _try_file_open(os.path.join(folder, "init",
                                             "control." + str(control) + ".init")) as outfile:
                outfile.writelines(all_lines)

        for state in range(18):
            all_lines = [
                "#Starting point file\n",
                "# This file contains the values of the initial points\n",
                "# for variable state #{0}\n".format(state), "\n", "# Type of initialization :\n",
                "linear\n", "\n", "# Number of interpolation points :\n",
                "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

            for i, time in enumerate(params['times']):
                all_lines.append("{0} {1}\n".format(time,
                                                    np.round(init.state(time)[state], decimals=2)))

            with _try_file_open(os.path.join(folder, "init", "state."+str(state)+".init")) as outfile:
                outfile.writelines(all_lines)

        all_lines = [
            "#Starting point file\n",
            "# This file contains the values of the initial points\n",
            "# for variable state #18\n", "\n", "# Type of initialization :\n",
            "linear\n", "\n", "# Number of interpolation points :\n",
            "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

        for i, time in enumerate(params['times']):
            state_val = integrate.simps(
                np.sum([init.state(t)[1:-1:3] for t in params['times'][:(i+1)]], axis=1),
                x=params['times'][:(i+1)])
            all_lines.append("{0} {1}\n".format(time,
                                                np.round(state_val, decimals=2)))

        with _try_file_open(os.path.join(folder, "init", "state.18.init")) as outfile:
            outfile.writelines(all_lines)
    else:
        raise RuntimeError("Must provide either a warm start solution file, or a ModelRun!")


def _try_file_open(filename):
    """Try repeatedly opening file for writing when permission errors occur in OneDrive."""

    while True:
        try:
            return open(filename, "w")
        except PermissionError:
            print("Permission error opening {0}. Trying again...".format(filename))
            continue
        break
