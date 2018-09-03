"""Analyse lag between deterministic and stochastic analogues of the same spatial model."""

import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

import simulation, space_model, fit_space_model, visualisation

plt.style.use("seaborn-whitegrid")

def get_lag_times(coupling, nruns=1000, keep_sims=False):
    """Setup stochastic and deterministic analogues and evaluate lag times to regions B and C.

    keep_sims: If True then all simulation data and deterministic run data is returned also.
    """

    initial_state_a = [495, 5, 0, 0, 0, 0, 0, 0]
    initial_state_b = [500, 0, 0, 0, 0, 0, 0, 0]
    initial_state_c = [500, 0, 0, 0, 0, 0, 0, 0]

    ######################
    ## STOCHASTIC MODEL ##
    ######################

    nodes = []
    nodes.append(simulation.Node((0, 0), "A", initial_state_a, 0))
    nodes.append(simulation.Node((0, 0), "B", initial_state_b, 1))
    nodes.append(simulation.Node((0, 0), "C", initial_state_c, 2))

    beta = 0.03

    risk_coupling = np.ones((2, 2))
    dist_coupling = np.array([[beta, coupling, 0],
                              [coupling, beta, coupling],
                              [0, coupling, beta]])

    # Setup required parameters
    params = {
        "birth_rate": 0.0,
        "death_rate": 0.0,
        "removal_rate": 1.0,
        "recov_rate": 0.0,
        "end_time": 50.0,
        "update_control_on_all_events": True,
        "control_update_freq": np.inf,
        "return_event_data": False
    }

    simulator = simulation.Simulator(nodes, risk_coupling, dist_coupling, params, None)
    stoch_run_data = simulator.run_simulation(nruns=nruns, verbose=False)

    stoch_time_max = np.zeros((nruns, 3))
    for j, run in enumerate(stoch_run_data):
        stoch_time_max[j, 0] = run.run_data["RegionA"][
            np.argmax(np.array(run.run_data["RegionA"])[:, 2])][0]

        arg_b = np.argmax(np.array(run.run_data["RegionB"])[:, 2])
        if run.run_data["RegionB"][arg_b][2] > 5:
            stoch_time_max[j, 1] = run.run_data["RegionB"][arg_b][0]
        else:
            stoch_time_max[j, 1] = np.nan

        arg_c = np.argmax(np.array(run.run_data["RegionC"])[:, 2])
        if run.run_data["RegionC"][arg_c][2] > 5:
            stoch_time_max[j, 2] = run.run_data["RegionC"][arg_c][0]
        else:
            stoch_time_max[j, 2] = np.nan

    #########################
    ## DETERMINISTIC MODEL ##
    #########################

    model_params = {
        "birth_rate": 0.0,
        "death_rate": 0.0,
        "removal_rate": 1.0,
        "recov_rate": 0.0,
        "state_init": np.array([495, 5, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0],
                               dtype=float),
        'times': np.linspace(0.0, params['end_time'], 2001),
        'max_control_rate': 0.0,
        'high_alloc_cost': 0.0,
        'low_alloc_cost': 0.0
    }

    space_fitter = fit_space_model.SpaceFitterLikelihood(None)
    space_fitter.data['sigma'] = dist_coupling
    space_fitter.data['rho'] = risk_coupling
    determ_model = space_model.SpaceModel(model_params, space_fitter)
    determ_run_data = determ_model.run_policy(space_model.no_control_policy)

    determ_time_max = np.array([model_params['times'][np.argmax([
        determ_run_data.state(t)[6*x+1] for t in model_params['times']])] for x in range(3)])

    lag_times = stoch_time_max - determ_time_max

    if keep_sims:
        return (lag_times, [run.run_data for run in stoch_run_data], determ_run_data)
    return (lag_times,)

def make_data(nruns=1000):
    """Generate data for later plotting."""

    all_couplings = np.geomspace(10**-7, 1, 51)
    sample_couplings = [10**-4, 10**-5]
    all_times = []

    for coupling in all_couplings:
        lag_times, = get_lag_times(coupling, nruns=nruns)
        all_times.append(lag_times)

        print("Done coupling {}".format(coupling))

    np.savez(os.path.join("Data", "lag_times"), couplings=all_couplings, lag_times=all_times)

    sim_data = {
        'stoch_sims': [],
        'determ_runs': []
    }
    for sample_coupling in sample_couplings:
        _, stoch_run_data, determ_run_data = get_lag_times(
            sample_coupling, nruns=nruns, keep_sims=True)
        sim_data['stoch_sims'].append(stoch_run_data)
        sim_data['determ_runs'].append(determ_run_data)

    with open(os.path.join("Data", "lag_data_runs.pickle"), "wb") as outfile:
        pickle.dump(sim_data, outfile)

def make_plot():
    """Make plot for supplementary material (Figure S7)."""

    initial_state_a = [495, 5, 0, 0, 0, 0, 0, 0]
    initial_state_b = [500, 0, 0, 0, 0, 0, 0, 0]
    initial_state_c = [500, 0, 0, 0, 0, 0, 0, 0]
    nodes = []
    nodes.append(simulation.Node((0, 0), "A", initial_state_a, 0))
    nodes.append(simulation.Node((0, 0), "B", initial_state_b, 1))
    nodes.append(simulation.Node((0, 0), "C", initial_state_c, 2))

    data = np.load(os.path.join("Data", "lag_times.npz"))
    means = np.nanmean(data['lag_times'], axis=1)
    errors = np.nanpercentile(data['lag_times'], [5, 95], axis=1)
    errors[0] = means - errors[0]
    errors[1] -= means

    for i, mean in enumerate(means):
        nonnan = [x for x in data['lag_times'][i, :, 1] if not np.isnan(x)]
        if len(nonnan) < 0.05*len(data['lag_times'][i, :, 1]):
            mean[0] = np.nan

        nonnan = [x for x in data['lag_times'][i, :, 2] if not np.isnan(x)]
        if len(nonnan) < 0.05*len(data['lag_times'][i, :, 2]):
            mean[1] = np.nan

    with open(os.path.join("Data", "lag_data_runs.pickle"), "rb") as infile:
        run_data = pickle.load(infile)

    dpc_times = np.linspace(0, 10.0, 101)
    stoch_dpc_data1 = visualisation._get_dpc_data(run_data['stoch_sims'][0], dpc_times, nodes)
    stoch_dpc_data2 = visualisation._get_dpc_data(run_data['stoch_sims'][1], dpc_times, nodes)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    fig = plt.figure()
    gs = gridspec.GridSpec(7, 4, height_ratios=[3, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5],
                           width_ratios=[1, 1, 1, 0.2])
    gs.update(top=0.95, left=0.1, right=0.95, hspace=2.5, wspace=0.35, bottom=0.1)
    dummy_axis1 = fig.add_subplot(gs[1, :-1], frameon=False)
    dummy_axis1.set_axis_off()
    dummy_axis2 = fig.add_subplot(gs[4, :-1], frameon=False)
    dummy_axis2.set_axis_off()
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.errorbar(data['couplings']*0.98, means[:, 1], errors[:, :, 1], label="Region B", fmt="o",
                 ms=1, lw=0.7)
    ax1.errorbar(data['couplings']*1.02, means[:, 2], errors[:, :, 2], label="Region C", fmt="o",
                 ms=1, lw=0.7)
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Coupling ($\epsilon$)", fontsize=10)
    ax1.set_ylabel("Lag", fontsize=10)
    ax1.set_xlim([10e-7, 1])

    dummy_axis1.text(0.5, 0.5, '\n' + r'High Coupling ($\epsilon = 10^{-4}$)',
                     horizontalalignment='center', verticalalignment='center',
                     transform=dummy_axis1.transAxes, fontsize=12)
    dummy_axis2.text(0.5, 0.5, '\n' + r'Low Coupling ($\epsilon = 10^{-5}$)',
                     horizontalalignment='center', verticalalignment='center',
                     transform=dummy_axis2.transAxes, fontsize=12)

    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.set_title("Region A", fontsize=8)
    ax3 = fig.add_subplot(gs[2:4, 1], sharey=ax2)
    ax3.set_title("Region B", fontsize=8)
    ax4 = fig.add_subplot(gs[2:4, 2], sharey=ax2)
    ax4.set_title("Region C", fontsize=8)

    ax_cbar = fig.add_subplot(gs[3:6, 3])
    cmap = mpl.colors.ListedColormap(
        [[1, 0, 0, x] for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1]])
    bounds = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm, boundaries=bounds,
                                     ticks=np.arange(10, 100, 20), spacing='proportional',
                                     orientation='vertical')
    cbar.add_lines([50], [(1, 0, 0)], 1)
    cbar.lines[0].set_dashes("dashed")
    cbar.set_label("Distribution Deciles", fontsize=8)
    ax_cbar.yaxis.set_label_position("left")

    visualisation.err_bands(stoch_dpc_data1['RegionA'][1], xvals=dpc_times, ax=ax2)
    visualisation.err_bands(stoch_dpc_data1['RegionB'][1], xvals=dpc_times, ax=ax3)
    visualisation.err_bands(stoch_dpc_data1['RegionC'][1], xvals=dpc_times, ax=ax4)

    ax2.plot(dpc_times, [run_data['determ_runs'][0].state(t)[1] for t in dpc_times], 'k--')
    ax3.plot(dpc_times, [run_data['determ_runs'][0].state(t)[7] for t in dpc_times], 'k--')
    ax4.plot(dpc_times, [run_data['determ_runs'][0].state(t)[13] for t in dpc_times], 'k--')

    ax5 = fig.add_subplot(gs[5:, 0])
    ax5.set_title("Region A", fontsize=8)
    ax6 = fig.add_subplot(gs[5:, 1], sharey=ax5)
    ax6.set_title("Region B", fontsize=8)
    ax7 = fig.add_subplot(gs[5:, 2], sharey=ax5)
    ax7.set_title("Region C", fontsize=8)

    visualisation.err_bands(stoch_dpc_data2['RegionA'][1], xvals=dpc_times, ax=ax5)
    visualisation.err_bands(stoch_dpc_data2['RegionB'][1], xvals=dpc_times, ax=ax6)
    visualisation.err_bands(stoch_dpc_data2['RegionC'][1], xvals=dpc_times, ax=ax7)

    ax5.plot(dpc_times, [run_data['determ_runs'][1].state(t)[1] for t in dpc_times], 'k--')
    ax6.plot(dpc_times, [run_data['determ_runs'][1].state(t)[7] for t in dpc_times], 'k--')
    ax7.plot(dpc_times, [run_data['determ_runs'][1].state(t)[13] for t in dpc_times], 'k--')

    for ax in [ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xlim([0, 10])
        ax.set_xlabel("Time", fontsize=10)
    ax2.set_ylabel(r"$I(t)$", fontsize=10)
    ax5.set_ylabel(r"$I(t)$", fontsize=10)

    fig.text(0.01, 0.96, "(a)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.66, "(b)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.32, "(c)", transform=fig.transFigure, fontsize=11, fontweight="semibold")

    fig.savefig(os.path.join("Figures", "SuppFig7.pdf"))

if __name__ == "__main__":
    make_data(nruns=1000)
    make_plot()
