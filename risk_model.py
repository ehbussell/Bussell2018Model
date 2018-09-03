"""Risk structured OCT model of network simulation, with optimisation methods."""

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

    return [0, 0]

def even_control_policy(time):
    """Policy carrying out no disease management."""

    return [0.5, 0.5]

class IncFunc:
    """Infection incidence function."""

    def __init__(self, fitter):
        """Make incidence function from fitter object."""

        self.predict = fitter.predict_rates
        self.beta = fitter.data.get('beta', None)

    def __call__(self, state):
        rates = np.ma.filled(self.predict(state))

        if rates.shape[1] == 4:
            high_rate = rates[0, 0] + rates[0, 1]
            low_rate = rates[0, 2] + rates[0, 3]
        elif rates.shape[1] == 2:
            high_rate = rates[0, 0]
            low_rate = rates[0, 1]
        else:
            raise ValueError("Incorrect rate array size!")

        return high_rate, low_rate

class RiskModel:
    """
    Class to implement risk structured OCT model.

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
    and a RiskFitter object.
    """

    def __init__(self, params, risk_fitter):
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

        self.inc_fn = IncFunc(risk_fitter)
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

        Control should give vaccination rate proportion for each risk as fn of time.
        """

        control = self._controller(time)
        sus_h, inf_h, vac_h, sus_l, inf_l, vac_l, obj = state

        high_inf_rate, low_inf_rate = self.inc_fn(np.array([[sus_h, inf_h, sus_l, inf_l]]))

        d_sus_h = (
            self.params['birth_rate'] * (sus_h + inf_h + vac_h)
            - self.params['death_rate'] * sus_h - high_inf_rate
            - self.params['max_control_rate'] * control[0] * sus_h / (sus_h + inf_h + vac_h)
            + self.params['recov_rate'] * inf_h)

        d_inf_h = (
            high_inf_rate - self.params['death_rate'] * inf_h - self.params['removal_rate'] * inf_h
            - self.params['recov_rate'] * inf_h)

        d_vac_h = (
            self.params['max_control_rate'] * control[0] * sus_h / (sus_h + inf_h + vac_h) -
            self.params['death_rate'] * vac_h)

        d_sus_l = (
            self.params['birth_rate'] * (sus_l + inf_l + vac_l)
            - self.params['death_rate'] * sus_l - low_inf_rate
            - self.params['max_control_rate'] * control[1] * sus_l / (sus_l + inf_l + vac_l)
            + self.params['recov_rate'] * inf_l)

        d_inf_l = (
            low_inf_rate - self.params['death_rate'] * inf_l - self.params['removal_rate'] * inf_l
            - self.params['recov_rate'] * inf_l)

        d_vac_l = (
            self.params['max_control_rate'] * control[1] * sus_l / (sus_l + inf_l + vac_l) -
            self.params['death_rate'] * vac_l)

        d_obj = (inf_h + inf_l + self.params['high_alloc_cost'] * control[0] +
                 self.params['low_alloc_cost'] * control[1])

        d_state = [d_sus_h, d_inf_h, d_vac_h, d_sus_l, d_inf_l, d_vac_l, d_obj]

        return d_state

    def run_policy(self, control_policy):
        """Run forward simulation using a given control policy.

        Control should give vaccination rate proportion for each risk as fn of time."""

        self._controller = control_policy

        initial_ode_value = np.zeros(7)
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

        self._controller = None

        return ModelRun(state=state_t, costate=None, control=control_t, objective=ode.y[-1])

    def run_bocop(self, bocop_dir=None, verbose=True, init_policy=None):
        """Run BOCOP solver and return optimal state, co-state and control.

        Returned control function gives vaccination rate proportion for each risk as fn of time.
        """

        if bocop_dir is None:
            bocop_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RiskModelBOCOP")

        if init_policy is None:
            initialisation = self.run_policy(no_control_policy)
        else:
            initialisation = self.run_policy(init_policy)
        set_bocop_params(self.params, self.inc_fn, init=initialisation, folder=bocop_dir)

        if verbose is True:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")], cwd=bocop_dir)
        else:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")],
                           cwd=bocop_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        state_t, costate_t, control_t, exit_text = bocop_utils.read_sol_file(
            os.path.join(bocop_dir, "problem.sol"), ignore_fail=True)

        objective_value = state_t(self.params['times'][-1])[-1]

        return ModelRun(state=state_t, costate=costate_t, control=control_t,
                        objective=objective_value, exit_code=exit_text)

def set_bocop_params(params, inc_fn, init=None, folder="RiskModelBOCOP"):
    """Save parameters and initial conditions to file for BOCOP optimisation.

    Optionally provide a ModelRun object to initialise from."""

    with open(os.path.join(folder, "problem.bounds"), "r") as infile:
        all_lines = infile.readlines()

    # Initial conditions
    all_lines[9] = str(params['state_init'][0]) + " " + str(params['state_init'][0]) + " equal\n"
    all_lines[10] = str(params['state_init'][1]) + " " + str(params['state_init'][1]) + " equal\n"
    all_lines[11] = str(params['state_init'][2]) + " " + str(params['state_init'][2]) + " equal\n"
    all_lines[12] = str(params['state_init'][3]) + " " + str(params['state_init'][3]) + " equal\n"
    all_lines[13] = str(params['state_init'][4]) + " " + str(params['state_init'][4]) + " equal\n"
    all_lines[14] = str(params['state_init'][5]) + " " + str(params['state_init'][5]) + " equal\n"
    all_lines[15] = "0 0 equal\n"

    # Max budget expenditure
    all_lines[35] = "-2e+020 1.0 upper\n"

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
    all_lines[10] = str(inc_fn.beta[0, 0]) + "\n"
    all_lines[11] = str(inc_fn.beta[0, 1]) + "\n"
    all_lines[12] = str(inc_fn.beta[1, 0]) + "\n"
    all_lines[13] = str(inc_fn.beta[1, 1]) + "\n"
    all_lines[14] = str(inc_fn.powers[0, 0]) + "\n"
    all_lines[15] = str(inc_fn.powers[1, 0]) + "\n"
    all_lines[16] = str(inc_fn.powers[2, 0]) + "\n"
    all_lines[17] = str(inc_fn.powers[3, 0]) + "\n"
    all_lines[18] = str(inc_fn.powers[0, 1]) + "\n"
    all_lines[19] = str(inc_fn.powers[1, 1]) + "\n"
    all_lines[20] = str(inc_fn.powers[2, 1]) + "\n"
    all_lines[21] = str(inc_fn.powers[3, 1]) + "\n"
    all_lines[22] = str(params['high_alloc_cost']) + "\n"
    all_lines[23] = str(params['low_alloc_cost']) + "\n"

    with _try_file_open(os.path.join(folder, "problem.constants")) as outfile:
        outfile.writelines(all_lines)

    with open(os.path.join(folder, "problem.def"), "r") as infile:
        all_lines = infile.readlines()

    n_steps = str(len(params['times']) - 1)
    all_lines[5] = "time.initial double " + str(params['times'][0]) + "\n"
    all_lines[6] = "time.final double " + str(params['times'][-1]) + "\n"
    all_lines[18] = "discretization.steps integer " + n_steps + "\n"

    with _try_file_open(os.path.join(folder, "problem.def")) as outfile:
        outfile.writelines(all_lines)

    # Initialisation
    control_init = np.array([[init.control(t)[j] for j in range(2)] for t in params['times']])

    for control in range(2):
        all_lines = [
            "#Starting point file\n",
            "# This file contains the values of the initial points\n",
            "# for variable control #{0}\n".format(control), "\n", "# Type of initialization :\n",
            "linear\n", "\n", "# Number of interpolation points :\n",
            "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

        for i, time in enumerate(params['times']):
            all_lines.append("{0} {1}\n".format(time,
                                                np.round(control_init[i, control], decimals=2)))

        with _try_file_open(os.path.join(folder, "init",
                                         "control." + str(control) + ".init")) as outfile:
            outfile.writelines(all_lines)

    for state in range(6):
        all_lines = [
            "#Starting point file\n",
            "# This file contains the values of the initial points\n",
            "# for variable state #{0}\n".format(state), "\n", "# Type of initialization :\n",
            "linear\n", "\n", "# Number of interpolation points :\n",
            "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

        for i, time in enumerate(params['times']):
            all_lines.append("{0} {1}\n".format(time,
                                                np.round(init.state(time)[state], decimals=1)))

        with _try_file_open(os.path.join(folder, "init", "state."+str(state)+".init")) as outfile:
            outfile.writelines(all_lines)

    all_lines = [
        "#Starting point file\n",
        "# This file contains the values of the initial points\n",
        "# for variable state #6\n", "\n", "# Type of initialization :\n",
        "linear\n", "\n", "# Number of interpolation points :\n",
        "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

    for i, time in enumerate(params['times']):
        state_val = integrate.simps(
            np.sum([init.state(t)[[1, 4]] for t in params['times'][:(i+1)]], axis=1),
            x=params['times'][:(i+1)])
        all_lines.append("{0} {1}\n".format(time,
                                            np.round(state_val, decimals=1)))

    with _try_file_open(os.path.join(folder, "init", "state.6.init")) as outfile:
        outfile.writelines(all_lines)

def _try_file_open(filename):
    """Try repeatedly opening file for writing when permission errors occur in OneDrive."""

    while True:
        try:
            return open(filename, "w")
        except PermissionError:
            print("Permission error opening {0}. Trying again...".format(filename))
            continue
        break
