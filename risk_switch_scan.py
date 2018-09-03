"""Generate Figure S11 scanning over risk switch time."""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed

import simulation, risk_model, fitting, control_tools, visualisation

def run_sims(switch_time, nruns, setup, model, params):
    """Run simulations and extract objective values."""

    def controller(time):
        if time < switch_time:
            return [1.0, 0.0]
        else:
            return [0.0, 1.0]

    params["controller_args"] = {
        "oc_model": model,
        "policy": controller
    }

    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'],
                                     setup['dist_coupling'], params,
                                     control_tools.RiskPolicyController)
    run_data = simulator.run_simulation(nruns=nruns, verbose=False)

    return [np.sum(x.objective) for x in run_data]

def make_data(switch_times, nruns):
    """Generate data for all switch times."""

    with open(os.path.join("Data", "data.pickle"), "rb") as infile:
        all_data = pickle.load(infile)

    fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))
    setup = all_data['setup']
    params = {
        "birth_rate": setup['birth_rate'],
        "death_rate": setup['death_rate'],
        "removal_rate": setup['removal_rate'],
        "recov_rate": setup['recov_rate'],
        "end_time": setup['end_time'],
        "update_control_on_all_events": True,
        "control_update_freq": np.inf,
        "return_event_data": False,
    }

    model_params = {
        'birth_rate': setup['birth_rate'],
        'death_rate': setup['death_rate'],
        'removal_rate': setup['removal_rate'],
        'recov_rate': setup['recov_rate'],
        'state_init': np.array(all_data['sim_no_control'][0][0]['Global'][0])[[1, 2, 3, 5, 6, 7]],
        'times': np.linspace(0, setup['end_time'], 101),
        'max_control_rate': setup['max_control_rate'],
        'high_alloc_cost': 0.0,
        'low_alloc_cost': 0.0
    }

    model = risk_model.RiskModel(model_params, fitter.get_fit())

    risk_run_data = Parallel(n_jobs=6)(delayed(run_sims)(
        switch_time, nruns, setup, model, params) for switch_time in switch_times)

    np.savez(os.path.join("Data", "SwitchOptimisation"),
             switch_times=switch_times, risk_run_data=risk_run_data)

def make_fig(data_file):
    """Create Figure S11."""

    plt.style.use("seaborn-whitegrid")

    data = np.load(data_file)
    switch_times = data['switch_times']
    risk_run_data = data['risk_run_data']

    order = np.argsort(switch_times)
    switch_times = switch_times[order]
    risk_run_data = np.array(risk_run_data)[order]

    cmap = mpl.colors.ListedColormap(
        [[1, 0, 0, x] for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1]])
    bounds = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_cbar = fig.add_axes([0.2, 0.78, 0.25, 0.05])
    visualisation.err_bands(risk_run_data, ax, xvals=switch_times, label="Median")
    ax.plot(switch_times, [np.mean(x) for x in risk_run_data], "b--", label="Mean")
    ax.set_xlabel("Risk Switch Time")
    ax.set_ylabel("Epidemic Cost")
    ax.set_ylim([-40, 1600])
    ax.legend(loc="best", ncol=2)
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=bounds,
                                     ticks=np.arange(10, 100, 20), spacing='proportional',
                                     orientation='horizontal')
    cbar.add_lines([50], [(1, 0, 0)], 1)
    cbar.lines[0].set_dashes("dashed")
    cbar.set_label("Distribution Deciles", fontsize=10)
    ax_cbar.xaxis.set_label_position("top")
    fig.savefig(os.path.join("Figures", "SuppFig11.pdf"), dpi=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

    with open(os.path.join("Data", "data.pickle"), "rb") as infile:
        all_data = pickle.load(infile)
    final_time = all_data['setup']['end_time']

    switch_times = np.linspace(0.0, final_time, 101)
    make_data(switch_times, nruns=1000)
    make_fig(os.path.join("Data", "SwitchOptimisation.npz"))
