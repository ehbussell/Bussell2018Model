"""Functions to help when using the BOCOP direct solver for optimal control problems."""

import os
import numpy as np
from scipy.interpolate import interp1d


def read_sol_file(file="problem.sol", ignore_fail=False):
    """Read BOCOP solution file and extract state, costate and control as interpolated functions."""

    with open(file, 'r') as infile:
        all_lines = infile.readlines()

    for i, line in enumerate(all_lines):
        if "time.initial" in line:
            time_init = float(line.split()[-1])
        if "time.final" in line:
            time_final = float(line.split()[-1])
        if "state.dimension" in line:
            state_dim = int(line.split()[-1])
        if "control.dimension" in line:
            control_dim = int(line.split()[-1])
        if "discretization.steps" in line:
            time_steps = int(line.split()[-1])

    times = np.linspace(time_init, time_final, time_steps + 1)

    Xt = []
    Lt = []
    Ut = []

    for i, line in enumerate(all_lines):
        if "# Number of stages of discretization method :" in line:
            times_control = list(map(float, all_lines[i+3:i+4+3*time_steps]))
            times_control = [time_init + x*(time_final - time_init)
                             for x in times_control]
            del times_control[::3]

        for j in range(state_dim):
            if "# State " + str(j) + "\n" == line:
                Xt.append(list(map(float, all_lines[i+1:i+2+time_steps])))

        for j in range(control_dim):
            if "# Control " + str(j) + "\n" == line:
                Ut.append(list(map(float, all_lines[i+1:i+1+2*time_steps])))

        for j in range(state_dim):
            if ("# Dynamic constraint " + str(j) +
                    " (y_dot - f = 0) multipliers :") in line:
                Lt.append(list(map(float, all_lines[i+1:i+1+time_steps])))

    Xt = interp1d(times, Xt, fill_value="extrapolate")          # State
    Lt = interp1d(times[:-1], Lt, fill_value="extrapolate")     # Costate
    Ut = interp1d(times_control, Ut, fill_value="extrapolate")  # Control

    results_file = os.path.join(os.path.dirname(file), "result.out")
    with open(results_file, "r") as infile:
        result_lines = infile.readlines()

    exit_text = None
    for line in result_lines:
        if "EXIT" in line:
            exit_text = line[6:].strip()

    if exit_text != "Optimal Solution Found." and not ignore_fail:
        raise RuntimeError("BOCOP optimisation failed with code: {0}".format(exit_text))

    return (Xt, Lt, Ut, exit_text)
