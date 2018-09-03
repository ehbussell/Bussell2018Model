"""Generate Figure S8 scanning over risk group control proportion."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed

import simulation, risk_model, control_tools, visualisation

def make_data(fitter, setup, state_init, num_props, nruns):
    """Generate all run data over range of risk proportions."""

    high_risk_proportions = np.linspace(0, 1, num_props)

    risk_run_data = Parallel(n_jobs=6)(delayed(get_objective_set)(
        fitter, setup, state_init, high_prop, nruns) for high_prop in high_risk_proportions)

    np.savez(os.path.join("Data", "RiskOptimisation"),
             high_risk_props=high_risk_proportions, risk_run_data=risk_run_data)

    min_prop_idx = np.argmin([np.mean(x) for x in risk_run_data])
    min_prop = high_risk_proportions[min_prop_idx]

    return min_prop

def get_objective_set(fitter, setup, state_init, high_prop, nruns):
    """Get objective values from a number of simulation runs using given control proportion."""

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
        'state_init': state_init,
        'times': np.linspace(0, setup['end_time'], 101),
        'max_control_rate': setup['max_control_rate'],
        'high_alloc_cost': 0.0,
        'low_alloc_cost': 0.0
    }

    model = risk_model.RiskModel(model_params, fitter.get_fit())

    def controller(time):
        return [high_prop, 1.0 - high_prop]

    params["controller_args"] = {
        "oc_model": model,
        "policy": controller
    }

    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'],
                                     setup['dist_coupling'], params,
                                     control_tools.RiskPolicyController)
    run_data = simulator.run_simulation(nruns=nruns, verbose=False)
    print("Done high proportion {}".format(high_prop))
    return [np.sum(x.objective) for x in run_data]

def make_fig(data_file):
    """Create Figure S8 from ."""

    plt.style.use("seaborn-whitegrid")

    data = np.load(data_file)
    high_risk_proportions = data['high_risk_props']
    risk_run_data = data['risk_run_data']

    min_prop_idx = np.argmin([np.mean(x) for x in risk_run_data])
    min_prop = high_risk_proportions[min_prop_idx]

    cmap = mpl.colors.ListedColormap(
        [[1, 0, 0, x] for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1]])
    bounds = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_cbar = fig.add_axes([0.62, 0.78, 0.25, 0.05])
    visualisation.err_bands(risk_run_data, ax, xvals=high_risk_proportions, label="Median")
    ax.plot(high_risk_proportions, [np.mean(x) for x in risk_run_data], "b--", label="Mean")
    ax.axvline(min_prop, ls="--", color="black", alpha=0.5)
    ax.set_xlabel("High Risk Proportion")
    ax.set_ylabel("Epidemic Cost")
    ax.legend(loc="best", ncol=2)
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=bounds,
                                     ticks=np.arange(10, 100, 20), spacing='proportional',
                                     orientation='horizontal')
    cbar.add_lines([50], [(1, 0, 0)], 1)
    cbar.lines[0].set_dashes("dashed")
    cbar.set_label("Distribution Deciles", fontsize=10)
    ax_cbar.xaxis.set_label_position("top")
    fig.savefig(os.path.join("Figures", "SuppFig8.pdf"), dpi=600)

if __name__ == "__main__":
    make_fig(os.path.join("Data", "RiskOptimisation.npz"))
