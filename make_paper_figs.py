"""Generate figures for paper."""

import copy
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl

import simulation, risk_model, fitting, control_tools, visualisation, space_model
from simulation import State
import risk_split_scan

def make_data(nruns=10, reuse_fitter=False, skip_risk_mpc=False, skip_space_mpc=False,
              check_data=None):
    """Generate all data for figures.

    Arguments:
        nruns           Number of simulation runs for each control strategy
        reuse_fitter    Whether to reuse previous fits. If so take from Data/Fit/FitData.pickle
        skip_risk_mpc   If True do not run risk based MPC simulations
        skip_space_mpc  If True do not run space based MPC simulations
        check_data      If not None this is filepath for existing data that this new data will be
                        appended to. Setup in each will be checked to ensure parameters have not
                        been changed.
    """

    plt.style.use("seaborn-whitegrid")

    if check_data is not None:
        with open(check_data, "rb") as infile:
            setup_check = pickle.load(infile)['setup']

    all_data = {}

    # Setup initial network structure and state from node file
    node_file = "node_file.txt"
    nodes = simulation.initialise_nodes(node_file)
    nodes = fitting.randomise_infection(nodes, nlow=3, nhigh=0, node_choice=0)

    # Setup spatial and risk coupling
    beta = 2.5
    scale = 0.2
    risk_coupling = np.array([[1, 0.008], [0.008, 0.016]])
    dist_coupling = np.zeros((len(nodes), len(nodes)))
    np.fill_diagonal(dist_coupling, 1.0)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1.region == node2.region:
                # within region connections
                dist = np.linalg.norm(node1.position - node2.position)
                dist_coupling[i, j] = np.exp(-dist / scale)

    for k, l in [(17, 20), (18, 21), (19, 22), (32, 52), (33, 53), (34, 54)]:
        # between region connections
        dist_coupling[l, k] = 0.1
        dist_coupling[k, l] = 0.1
    for i, node1 in enumerate(nodes):
        dist_coupling[:, i] *= beta

    # Setup required parameters for simulations and fitting
    setup = {
        'nodes': nodes,
        'dist_coupling': dist_coupling,
        'risk_coupling': risk_coupling,
        'birth_rate': 0.01,
        'death_rate': 0.01,
        'removal_rate': 0.5,
        'recov_rate': 0.25,
        'end_time': 5,
        'max_control_rate': 200,
        'mpc_update_freq': 0.5,
        'bounds': {
            'beta': np.array([[(0, 1), (0, 1)], [(0, 1), (0, 1)]]),
            'coef': np.full((4, 2, 2), (0, 3), dtype=np.float),
            'sigma': np.array([[(0, 1), (0, 1), (0, 1)],
                               [(0, 1), (0, 1), (0, 1)],
                               [(0, 1), (0, 1), (0, 1)]]),
            'rho': np.array([[(0, 1), (0, 1)],
                             [(0, 1), (0, 1)]])
        },
        'start': {
            'beta': np.array([[8.8e-3, 5.8e-5], [7.7e-4, 6.2e-4]]),
            'coef': np.array([[1.9, 1.0], [1.5, 1.0],
                              [2.4, 1.0], [1.9, 1.1]]),
            'sigma': np.array([[5.4e-2, 4.9e-5, 0],
                               [7.7e-5, 5.3e-2, 1.5e-4],
                               [0, 1.5e-4, 5.5e-2]]),
            'rho': np.array([[1, 0.03],
                             [0.03, 0.05]])

        },
    }

    all_data['setup'] = setup

    params = {
        "birth_rate": setup['birth_rate'],
        "death_rate": setup['death_rate'],
        "removal_rate": setup['removal_rate'],
        "recov_rate": setup['recov_rate'],
        "end_time": setup['end_time'],
        "update_control_on_all_events": True,
        "control_update_freq": None,
        "return_event_data": True
    }

    os.makedirs(os.path.join("Data", "Fit"), exist_ok=True)

    # Test run and show sample no control simulations
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, None)
    no_control_run_data = simulator.run_simulation(nruns=5)
    visualisation.plot_dpc_data(nodes, no_control_run_data, {}, nruns=5)
    plt.show()

    # Run fitting of approximate models
    if reuse_fitter:
        fitter = fitting.Fitter.from_file(os.path.join("Data", "Fit", "FitData.pickle"))
    else:
        fitter = fitting.Fitter(nodes, risk_coupling, dist_coupling, params)
        fitter.get_data(initial_nodes=nodes)
        fitter.fit_risk(setup['bounds'], setup['start'])
        fitter.fit_space(setup['bounds'], setup['start'])

        fitter.save(filename=os.path.join("Data", "Fit", "FitData.pickle"))
        print(fitter)
        fitter.assess(save_folder=os.path.join("Data", "Fit"), initial_nodes=nodes,
                      max_control_rate=setup['max_control_rate'])
        fitter.save(filename=os.path.join("Data", "Fit", "FitData.pickle"))

    params['return_event_data'] = False

    # No control simulation results
    print("No control simulation runs")
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, None)

    no_control_run_data = simulator.run_simulation(nruns=nruns)
    all_data['sim_no_control'] = no_control_run_data

    # Scenario testing runs
    model_params = {
        'birth_rate': setup['birth_rate'],
        'death_rate': setup['death_rate'],
        'removal_rate': setup['removal_rate'],
        'recov_rate': setup['recov_rate'],
        'state_init': np.array(no_control_run_data[0][0]['Global'][0])[[1, 2, 3, 5, 6, 7]],
        'times': np.linspace(0, setup['end_time'], 101),
        'max_control_rate': setup['max_control_rate'],
        'high_alloc_cost': 0.0,
        'low_alloc_cost': 0.0
    }
    space_model_params = copy.deepcopy(model_params)
    state = np.zeros(18)
    state[0:6] = np.array(no_control_run_data[0][0]["RegionA"][0])[[1, 2, 3, 5, 6, 7]]
    state[6:12] = np.array(no_control_run_data[0][0]["RegionB"][0])[[1, 2, 3, 5, 6, 7]]
    state[12:18] = np.array(no_control_run_data[0][0]["RegionC"][0])[[1, 2, 3, 5, 6, 7]]
    space_model_params['state_init'] = state

    if check_data is None:
        # Get optimal constant split strategy balancing high and low control
        print("Optimising constant risk split strategy")
        min_prop = risk_split_scan.make_data(
            fitter, setup, model_params['state_init'], num_props=101, nruns=1000)
        setup['risk_min_prop'] = min_prop
    else:
        setup['risk_min_prop'] = setup_check['risk_min_prop']

    # Check setup is same as any previous data
    if check_data is not None:
        for key, val in setup.items():
            if isinstance(val, dict):
                for key2 in val.keys():
                    if np.any(val[key2] != setup_check[key][key2]):
                        raise ValueError("Setup parameters do not match in {0}!".format(key))
            else:
                if np.any(val != setup_check[key]):
                    raise ValueError("Setup parameters do not match in {0}!".format(key))

    # Approximate models
    model = risk_model.RiskModel(model_params, fitter.get_fit())
    model_space = space_model.SpaceModel(space_model_params, fitter.get_fit(space=True))

    def high_prio_controller(time):
        return [1.0, 0.0]

    def split_prio_controller(time):
        return [setup['risk_min_prop'], 1.0 - setup['risk_min_prop']]

    params["controller_args"] = {
        "oc_model": model,
        "policy": high_prio_controller
    }
    # Prioritise high risk population
    print("High risk prioritisation simulation runs")
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, control_tools.RiskPolicyController)
    high_run_data = simulator.run_simulation(nruns=nruns)
    all_data['sim_high'] = high_run_data

    # Constant optimal risk split population
    print("Constant risk split simulation runs")
    params["controller_args"]["policy"] = split_prio_controller
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, control_tools.RiskPolicyController)
    split_run_data = simulator.run_simulation(nruns=nruns)
    all_data['sim_split'] = split_run_data

    # Risk based model - no control DPC
    no_control_risk_run = model.run_policy(risk_model.no_control_policy)
    all_data['risk_model_no_control'] = no_control_risk_run

    # Risk based model - optimal control
    print("Risk model open loop simulation runs")
    bocop_run = model.run_bocop(verbose=True, init_policy=risk_model.even_control_policy)
    if bocop_run.exit_code != "Optimal Solution Found.":
        raise RuntimeError("Convergence Failure!")
    params["controller_args"]["policy"] = bocop_run.control
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, control_tools.RiskPolicyController)
    risk_opt_run_data = simulator.run_simulation(nruns=nruns)
    all_data['sim_risk_opt'] = risk_opt_run_data
    all_data['risk_model_opt'] = bocop_run

     # Space based model - no control DPC
    no_control_space_run = model_space.run_policy(space_model.no_control_policy)
    all_data['space_model_no_control'] = no_control_space_run

    # Space based model - optimal control
    print("Space model open loop simulation runs")
    params['vac_rate'] = 1.0
    bocop_run = model_space.run_bocop(verbose=True, sol_file="InitSol0.sol")
    if bocop_run.exit_code != "Optimal Solution Found.":
        raise RuntimeError("Convergence Failure!")
    params["controller_args"]["policy"] = bocop_run.control
    params["controller_args"]["oc_model"] = model_space
    simulator = simulation.Simulator(setup['nodes'], setup['risk_coupling'], setup['dist_coupling'],
                                     params, control_tools.SpacePolicyController)
    space_opt_run_data = simulator.run_simulation(nruns=nruns)
    all_data['sim_space_opt'] = space_opt_run_data
    all_data['space_model_opt'] = bocop_run

    if not skip_risk_mpc:
        # Risk based MPC runs
        print("Risk model MPC simulation runs")
        params['controller_args'] = {
            'oc_model': model,
            'mpc_params': {
                'update_freq': setup['mpc_update_freq'],
                'model_horizon': model_params['times'][-1],
                'rolling_horizon': False,
                'init_policy': all_data['risk_model_opt'].control,
                'initial_control_func': None
            },
            'verbose': True
        }
        params['control_update_freq'] = setup['mpc_update_freq']
        simulator = simulation.Simulator(
            setup['nodes'], setup['risk_coupling'], setup['dist_coupling'], params,
            control_tools.MPCRiskController)
        mpc_run_data = simulator.run_simulation(nruns=nruns)
        all_data['sim_risk_mpc'] = mpc_run_data

    if not skip_space_mpc:
        # Setup cold start intialisation files
        update_times = np.arange(0, setup['end_time'], setup['mpc_update_freq'])
        sol_file_names = ["InitSol" + str(i) + ".sol" for i in range(len(update_times))]
        if not reuse_fitter:
            for i, (sol_file, start_time) in enumerate(zip(sol_file_names[1:], update_times[1:])):
                model_space.params['state_init'] = all_data['space_model_opt'].state(
                    start_time)[:-1]
                # model_space.params['times'] = np.linspace(start_time, setup['end_time'], 101)
                model_space.params['times'] = np.arange(start_time, setup['end_time'],
                                                        setup['end_time'] / 100)
                bocop_run = model_space.run_bocop(verbose=True, sol_file=sol_file,
                                                  init_policy=space_model.even_control_policy)
                if bocop_run.exit_code != "Optimal Solution Found.":
                    raise RuntimeError("Convergence Failure!")

        # Space based MPC runs
        print("Space model MPC simulation runs")
        params['controller_args'] = {
            'oc_model': model_space,
            'mpc_params': {
                'update_freq': setup['mpc_update_freq'],
                'model_horizon': space_model_params['times'][-1],
                'rolling_horizon': False,
                'cold_start_files': sol_file_names,
                'init_policy': all_data['space_model_opt'].control,
                'initial_control_func': None
            },
            'verbose': True
        }
        params['control_update_freq'] = setup['mpc_update_freq']
        simulator = simulation.Simulator(
            setup['nodes'], setup['risk_coupling'], setup['dist_coupling'], params,
            control_tools.MPCSpaceController)
        mpc_run_data = simulator.run_simulation(nruns=nruns)
        all_data['sim_space_mpc'] = mpc_run_data

    params["controller_args"] = None
    return all_data

def make_plots(all_data, skip_risk_mpc=False, skip_space_mpc=False):
    """Generate figures from data for main paper."""

    os.makedirs(os.path.join("Figures", "Diagnostics"), exist_ok=True)

    viz_options = {
        'population_vmin': 0,
        'population_vmax': 100,
        'show_dpcs': True,
        'alpha': 0.2,
        'dpc_data': all_data['sim_no_control'][0:20],
        'regions': ["A", "B", "C"],
        'inf_vmax': 0.05,
        'inf_cmap': mpl.colors.ListedColormap(["orange", "red"]),
        'node_alpha': 0.75,
        'min_node_radius': 0.025,
        'max_node_radius': 0.075,
        'coupling_link_min_size': 0.5,
        'coupling_link_max_size': 1.0,
    }

    plt.style.use("seaborn-whitegrid")

    nodes = all_data['setup']['nodes']
    dist_coupling = all_data['setup']['dist_coupling']

    ########################
    ## Main Paper Figures ##
    ########################

    # Get objective values
    all_objectives = []
    all_names = []
    iqr_segments = []

    all_objectives.append([np.sum(x.objective) for x in all_data['sim_high']])
    all_names.append("High")
    all_objectives.append([np.sum(x.objective) for x in all_data['sim_split']])
    all_names.append("Split")
    iqr_segments.append([[[i, np.percentile(x, 25)], [i, np.percentile(x, 75)]]
                         for i, x in enumerate(all_objectives[0:2])])

    all_objectives.append([np.sum(x.objective) for x in all_data['sim_risk_opt']])
    all_names.append("Risk OL")
    if skip_risk_mpc:
        all_objectives.append(np.zeros(100))
    else:
        all_objectives.append([np.sum(x.objective) for x in all_data['sim_risk_mpc']])
    all_names.append("Risk MPC")
    iqr_segments.append([[[i+2, np.percentile(x, 25)], [i+2, np.percentile(x, 75)]]
                         for i, x in enumerate(all_objectives[2:4])])

    all_objectives.append([np.sum(x.objective) for x in all_data['sim_space_opt']])
    all_names.append("Space OL")
    if skip_space_mpc:
        all_objectives.append(np.zeros(100))
    else:
        all_objectives.append([np.sum(x.objective) for x in all_data['sim_space_mpc']])
    all_names.append("Space MPC")
    iqr_segments.append([[[i+4, np.percentile(x, 25)], [i+4, np.percentile(x, 75)]]
                         for i, x in enumerate(all_objectives[4:6])])

    all_objectives.append([np.sum(x.objective) for x in all_data['sim_no_control']])
    all_names.append("No Control")

    # Figure 2 - Network and OL/MPC schematic
    # Plot network structure, and simulated and predicted infection numbers for open-loop & MPC

    # Choose whether to plot spatial results or risk based results
    space = True
    if space:
        obj_idx = 4
        name = 'space'
    else:
        obj_idx = 2
        name = 'risk'

    # Get index of 95th percentile OL simulation
    ol_idx = np.where(all_objectives[obj_idx] == np.percentile(
        all_objectives[obj_idx], 60, interpolation="nearest"))[0][0]
    # Generate OL DPC time course
    sim_ol_times = np.array([
        x[0] for x in all_data['sim_' + name +'_opt'][ol_idx].run_data['Global']])
    sim_ol_inf_h = np.array(
        [x[1 + State.INF_H] for x in all_data['sim_' + name +'_opt'][ol_idx].run_data['Global']])
    sim_ol_inf_l = np.array(
        [x[1 + State.INF_L] for x in all_data['sim_' + name +'_opt'][ol_idx].run_data['Global']])

    pre_ol_times = np.linspace(0, all_data['setup']['end_time'], 501)
    pre_ol_inf_h = np.array([
        np.sum(all_data[name + '_model_opt'].state(t)[1:-1:6]) for t in pre_ol_times])
    pre_ol_inf_l = np.array([
        np.sum(all_data[name + '_model_opt'].state(t)[4:-1:6]) for t in pre_ol_times])

    # Get index of 95th percentile MPC simulation
    mpc_idx = np.where(all_objectives[obj_idx+1] == np.percentile(
        all_objectives[obj_idx+1], 60, interpolation="nearest"))[0][0]

    # Generate MPC DPC time course
    sim_mpc_times = np.array([
        x[0] for x in all_data['sim_' + name +'_mpc'][mpc_idx].run_data['Global']])
    sim_mpc_inf_h = np.array([
        x[1 + State.INF_H] for x in all_data['sim_' + name +'_mpc'][mpc_idx].run_data['Global']])
    sim_mpc_inf_l = np.array([
        x[1 + State.INF_L] for x in all_data['sim_' + name +'_mpc'][mpc_idx].run_data['Global']])

    pre_mpc_times = []
    pre_mpc_inf_h = []
    pre_mpc_inf_l = []
    update_times = np.arange(0, all_data['setup']['end_time'], all_data['setup']['mpc_update_freq'])
    for i, update_time in enumerate(update_times):
        times = np.linspace(update_time, update_time + all_data['setup']['mpc_update_freq'],
                            500 / len(update_times), endpoint=False)
        pre_mpc_times.append(times)
        pre_mpc_inf_h.append([
            np.sum(all_data['sim_' + name + '_mpc'][mpc_idx].control[i][1].state(t)[1:-1:6])
            for t in times])
        pre_mpc_inf_l.append([
            np.sum(all_data['sim_' + name + '_mpc'][mpc_idx].control[i][1].state(t)[4:-1:6])
            for t in times])

    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1])
    gs.update(top=0.95, left=0.1, right=0.95, hspace=0.55, wspace=0.4, bottom=0.2)
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, gs[0, 0], width_ratios=[6, 1])
    ax1 = fig.add_subplot(gs1[0])
    ax_leg1 = fig.add_subplot(gs1[1], frameon=False)
    ax_leg1.grid = False
    ax_leg1.set_xticks([])
    ax_leg1.set_yticks([])

    gs_bottom = gridspec.GridSpec(1, 1, top=0.1, bottom=0.02)
    ax_leg2 = fig.add_subplot(gs_bottom[:, :], frameon=False)
    ax_leg2.grid = False
    ax_leg2.set_xticks([])
    ax_leg2.set_yticks([])

    visualisation.plot_node_network(nodes, dist_coupling, options=viz_options, ax=ax1)

    ax1.text(2.5, 5.5, "A", fontsize=8, weight="semibold")
    ax1.text(5.5, 5.7, "B", fontsize=8, weight="semibold")
    ax1.text(5.5, 2.8, "C", fontsize=8, weight="semibold")
    ax1.set_aspect("equal", anchor="W")

    visualisation.plot_control(
        all_data['sim_' + name + '_mpc'][mpc_idx], all_data['setup']['end_time'],
        comparison=all_data[name + '_model_opt'].control, ax=ax2,
        comparison_args={'label': 'OL', 'linestyle': (0, (1, 1))}, colors=["red", "skyblue"],
        alpha=0.4)

    ax2.scatter(update_times, [-0.05]*len(update_times), s=10, facecolor="k",
                edgecolor="k", linewidth=0, zorder=10, clip_on=False)

    ax3.plot(sim_ol_times, sim_ol_inf_h+sim_ol_inf_l, color="red", linestyle="steps-post",
             label="Simulation", alpha=0.3)
    ax3.plot(pre_ol_times, pre_ol_inf_h+pre_ol_inf_l, '--', linewidth=1.0, color="red",
             label="Approximate Model")

    ax3.scatter([0], [0], s=10, facecolor="k", edgecolor="k", linewidth=0,
                label="Update Times", zorder=10, clip_on=False)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number Infected")
    ax3.set_title("Open-loop (OL)")

    ax4.plot(sim_mpc_times, sim_mpc_inf_h+sim_mpc_inf_l, color="red", linestyle="steps-post",
             alpha=0.3)

    for times, inf_h, inf_l in zip(pre_mpc_times, pre_mpc_inf_h, pre_mpc_inf_l):
        ax4.plot(times, np.array(inf_h)+np.array(inf_l), '--', linewidth=1.0, color="red")
    ax4.scatter(update_times, [0]*len(update_times), s=10, facecolor="k",
                edgecolor="k", linewidth=0, zorder=10, clip_on=False)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Number Infected")
    ax4.set_title("MPC")

    xlim = [0, 5]
    ylim = [0, ax3.get_ylim()[1]]

    fig.text(0.01, 0.96, "(a)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.5, 0.96, "(b)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.53, "(c)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.5, 0.53, "(d)", transform=fig.transFigure, fontsize=11, fontweight="semibold")

    ax2.set_xlim(xlim)
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_xticks(range(6))
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xticks(range(6))
    ax_leg1.legend(*ax2.get_legend_handles_labels(), loc="center right", frameon=True,
                   fontsize=6, handlelength=1.0)
    ax_leg2.legend(*ax3.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=True,
                   fontsize=8, handlelength=1.5)

    fig.savefig(os.path.join("Figures", "Figure2.pdf"), dpi=600)

    # Figure 3 - Illustrative model comparison of strategies
    # Violin plot showing distribution of epidemic costs for each strategy
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    viol1 = ax.violinplot(all_objectives[0:2], positions=[0, 1], showmeans=True, showmedians=True)
    viol2 = ax.violinplot(all_objectives[2:4], positions=[2, 3], showmeans=True, showmedians=True)
    viol3 = ax.violinplot(all_objectives[4:6], positions=[4, 5], showmeans=True, showmedians=True)

    legend_elements = []
    colours = ["C0", "C1", "C2"]
    for i, viol in enumerate([viol1, viol2, viol3]):
        viol['cmeans'].set_color("r")
        viol['cmeans'].set_zorder(30)
        viol['cmedians'].set_color("b")
        viol['cmedians'].set_zorder(30)
        viol['cbars'].set_segments(iqr_segments[i])
        viol['cbars'].set_color("k")
        viol['cbars'].set_alpha(0.3)
        viol['cmaxes'].set_alpha(0)
        viol['cmins'].set_alpha(0)
        for part in viol['bodies']:
            part.set_color(colours[i])

    legend_elements.append(mpl.lines.Line2D([0], [0], color=viol1['cmeans'].get_color()[0],
                                            lw=viol1['cmeans'].get_linewidth()[0], label='Mean'))
    legend_elements.append(mpl.lines.Line2D([0], [0], color=viol1['cmedians'].get_color()[0],
                                            lw=viol1['cmedians'].get_linewidth()[0],
                                            label='Median'))
    legend_elements.append(mpl.lines.Line2D([0], [0], color="k",
                                            lw=viol1['cbars'].get_linewidth()[0], alpha=0.3,
                                            label='IQR'))

    ax.set_xticks(range(6))
    ax.set_xticklabels(all_names)
    ax.set_xlabel("Control Strategy")
    ax.set_ylabel("Epidemic Cost")
    ax.set_ylim([-60, 1400])
    ax.legend(handles=legend_elements)
    fig.tight_layout()

    ax.annotate('User-defined', xy=(0.5, 1100), xytext=(0.5, 1300), fontsize=12, ha='center',
                va='bottom', arrowprops=dict(
                    arrowstyle='-[, widthB=3.5, lengthB=0.75', lw=2.0, color=colours[0]),
                color=colours[0], weight="bold")
    ax.annotate('Risk based', xy=(2.5, 750), xytext=(2.5, 950), fontsize=12, ha='center',
                va='bottom', arrowprops=dict(
                    arrowstyle='-[, widthB=3.5, lengthB=0.75', lw=2.0, color=colours[1]),
                color=colours[1], weight="bold")
    ax.annotate('Space based', xy=(4.5, 650), xytext=(4.5, 850), fontsize=12, ha='center',
                va='bottom', arrowprops=dict(
                    arrowstyle='-[, widthB=3.5, lengthB=0.75', lw=2.0, color=colours[2]),
                color=colours[2], weight="bold")

    fig.savefig(os.path.join("Figures", "Figure3.pdf"), dpi=600)

    plt.rc('axes', labelsize=12)

    ###########################
    ## Supplementary Figures ##
    ###########################

    viz_options['show_vac'] = False

    # Figure S2
    # Typical simulation model trajectories
    fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
        nodes, all_data['sim_no_control'][0:10], options=viz_options, nruns=10)
    fig.savefig(os.path.join("Figures", "SuppFig2.pdf"), dpi=600)

    # Figure S3
    # Risk model fit
    viz_options["show_regions"] = False
    times = np.linspace(0, all_data['setup']['end_time'], 101)
    fig, _, glob_axes = visualisation.plot_dpc_data(
        nodes, all_data['sim_no_control'][0:20], options=viz_options, nruns=20)
    glob_axes[0].plot(times, [all_data['risk_model_no_control'].state(t)[0] for t in times], 'g--',
                      lw=2)
    glob_axes[0].plot(times, [all_data['risk_model_no_control'].state(t)[1] for t in times], 'r--',
                      lw=2)
    glob_axes[1].plot(times, [all_data['risk_model_no_control'].state(t)[3] for t in times], 'g--',
                      lw=2)
    glob_axes[1].plot(times, [all_data['risk_model_no_control'].state(t)[4] for t in times], 'r--',
                      lw=2)
    fig.savefig(os.path.join("Figures", "SuppFig3.pdf"), dpi=600)

    # Figure S4
    # Space model fit
    viz_options["show_regions"] = True
    times = np.linspace(0, all_data['setup']['end_time'], 101)
    fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
        nodes, all_data['sim_no_control'][0:20], options=viz_options, nruns=20)

    for i in range(6):
        reg_axes[i].plot(times, [all_data['space_model_no_control'].state(t)[3*i] for t in times],
                         'g--', lw=2)
        reg_axes[i].plot(times, [all_data['space_model_no_control'].state(t)[3*i+1] for t in times],
                         'r--', lw=2)
    for risk in range(2):
        glob_axes[risk].plot(times, np.sum([all_data['space_model_no_control'].state(t)[(3*risk)::6]
                                            for t in times], axis=1), 'g--', lw=2)
        glob_axes[risk].plot(times, np.sum(
            [all_data['space_model_no_control'].state(t)[(3*risk+1)::6] for t in times], axis=1),
            'r--', lw=2)
    fig.savefig(os.path.join("Figures", "SuppFig4.pdf"), dpi=600)

    # Figure S8
    # Risk Split Scan
    risk_split_scan.make_fig(os.path.join("Data", "RiskOptimisation.npz"))

    # Figure S9
    # Comparison of optimal controls - risk vs space based
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.01, 1], width_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, 0], frameon=False)
    ax_legend.grid = False
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    leg_colours = [mpl.colors.to_rgba("C{}".format(i), alpha=alpha) for alpha in [0.7, 0.25]
                   for i in range(3)]

    visualisation.plot_control(all_data['sim_risk_opt'][0], 5, risk_based=True, ax=ax1,
                               colors=["red", "skyblue"], alpha=0.4)
    visualisation.plot_control(all_data['sim_space_opt'][0], 5, risk_based=True, ax=ax2,
                               colors=["red", "skyblue"], alpha=0.4)
    visualisation.plot_control(all_data['sim_space_opt'][0], 5, risk_based=False, ax=ax3,
                               regions=["A", "B", "C"], colors=leg_colours)

    ax_legend.legend(*ax1.get_legend_handles_labels(), loc='center', ncol=2, frameon=True,
                     fontsize=8, handlelength=1.5)
    ax3.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=True, fontsize=8,
               handlelength=1.5)

    ax1.set_title("Risk Based")
    ax2.set_title("Space Based")
    ax3.set_title("Space Based")

    ax1.set_xlim([0, 5])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_yticks(np.linspace(0, 1, 5))
    ax1.set_xticks(range(6))

    ax2.set_ylim([-0.05, 1.05])
    ax2.set_yticks(np.linspace(0, 1, 5))
    ax2.set_xticks(range(6))

    ax3.set_xlim([0, 5])
    ax3.set_ylim([-0.02, 1.02])
    ax3.set_yticks(np.linspace(0, 1, 6))
    ax3.set_xticks(range(6))

    fig.text(0.01, 0.96, "(a)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.45, "(b)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.42, 0.96, "(c)", transform=fig.transFigure, fontsize=11, fontweight="semibold")

    fig.tight_layout()
    fig.savefig(os.path.join("Figures", "SuppFig9.pdf"), dpi=600, bbox_inches='tight')

    # Figure S10
    # Histogram of illustrative model strategy results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_vals = sorted(all_objectives[-1])
    n_vals = np.arange(1, len(x_vals)+1) / np.float(len(x_vals))
    ax.step(x_vals, n_vals, label=all_names[-1], lw=1, color="k")
    ax.scatter([np.percentile(x_vals, 95)], [0], s=20, facecolor="None", edgecolor="k",
               linewidth=1, clip_on=False, zorder=15)
    for i in range(3):
        x_vals = sorted(all_objectives[2*i])
        n_vals = np.arange(1, len(x_vals)+1) / np.float(len(x_vals))
        ax.step(x_vals, n_vals, label=all_names[2*i], lw=1, color="C"+str(i), alpha=0.5)
        ax.scatter([np.percentile(x_vals, 95)], [0], s=20, facecolor="None", edgecolor="C"+str(i),
                   linewidth=1, alpha=0.5, clip_on=False, zorder=15)

        x_vals = sorted(all_objectives[2*i+1])
        n_vals = np.arange(1, len(x_vals)+1) / np.float(len(x_vals))
        ax.step(x_vals, n_vals, label=all_names[2*i+1], lw=1, color="C"+str(i))
        ax.scatter([np.percentile(x_vals, 95)], [0], s=20, facecolor="None", edgecolor="C"+str(i),
                   linewidth=1, clip_on=False, zorder=15)

    ax.legend(loc="center right", ncol=1)
    ax.set_xlabel("Epidemic Cost")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim([-0.05, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join("Figures", "SuppFig10.pdf"), dpi=600)

    ########################
    ## Diagnostic Figures ##
    ########################

    fig = visualisation.plot_node_network(nodes, dist_coupling, options=viz_options)
    fig.tight_layout()
    fig.savefig(os.path.join("Figures", "Diagnostics", "NetworkStructure.pdf"), dpi=600)

    fig, *_ = visualisation.plot_dpc_data(
        nodes, all_data['sim_high'][0:20], options=viz_options, nruns=20)
    fig.savefig(os.path.join("Figures", "Diagnostics", "HighControlDPC.png"), dpi=600)

    fig, *_ = visualisation.plot_dpc_data(
        nodes, all_data['sim_split'][0:20], options=viz_options, nruns=20)
    fig.savefig(os.path.join("Figures", "Diagnostics", "SplitControlDPC.png"), dpi=600)

    viz_options["show_regions"] = True
    fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
        nodes, all_data['sim_risk_opt'][0:20], options=viz_options, nruns=20)
    glob_axes[0].plot(times, [all_data['risk_model_opt'].state(t)[0] for t in times],
                      'g--', lw=2)
    glob_axes[0].plot(times, [all_data['risk_model_opt'].state(t)[1] for t in times],
                      'r--', lw=2)
    glob_axes[0].plot(times, [all_data['risk_model_opt'].state(t)[2] for t in times],
                      '--', color="purple", lw=2)
    glob_axes[1].plot(times, [all_data['risk_model_opt'].state(t)[3] for t in times],
                      'g--', lw=2)
    glob_axes[1].plot(times, [all_data['risk_model_opt'].state(t)[4] for t in times],
                      'r--', lw=2)
    glob_axes[1].plot(times, [all_data['risk_model_opt'].state(t)[5] for t in times],
                      '--', color="purple", lw=2)
    fig.savefig(os.path.join("Figures", "Diagnostics", "RiskModelOptControl.png"), dpi=600)

    if not skip_risk_mpc:
        viz_options["show_regions"] = True
        fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
            nodes, all_data['sim_risk_mpc'][0:20], options=viz_options, nruns=20)
        fig.savefig(os.path.join("Figures", "Diagnostics", "MPC_risk_DPC.png"), dpi=600)

    fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
        nodes, all_data['sim_space_opt'][0:20], options=viz_options, nruns=20)
    for i in range(6):
        reg_axes[i].plot(times, [all_data['space_model_opt'].state(t)[3*i] for t in times], 'g--',
                         lw=2)
        reg_axes[i].plot(times, [all_data['space_model_opt'].state(t)[3*i+1] for t in times], 'r--',
                         lw=2)
        reg_axes[i].plot(times, [all_data['space_model_opt'].state(t)[3*i+2] for t in times], '--',
                         color="purple", lw=2)
    for risk in range(2):
        glob_axes[risk].plot(times, [np.sum(all_data['space_model_opt'].state(t)[(3*risk):-1:6])
                                     for t in times], 'g--', lw=2)
        glob_axes[risk].plot(times, [np.sum(all_data['space_model_opt'].state(t)[(3*risk+1):-1:6])
                                     for t in times], 'r--', lw=2)
        glob_axes[risk].plot(times, [np.sum(all_data['space_model_opt'].state(t)[(3*risk+2):-1:6])
                                     for t in times], '--', color="purple", lw=2)
    fig.savefig(os.path.join("Figures", "Diagnostics", "SpaceModelOptControl.png"), dpi=600)

    if not skip_space_mpc:
        viz_options["show_regions"] = True
        fig, reg_axes, glob_axes = visualisation.plot_dpc_data(
            nodes, all_data['sim_space_mpc'][0:20], options=viz_options, nruns=20)
        fig.savefig(os.path.join("Figures", "Diagnostics", "MPC_space_DPC.png"), dpi=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-a", "--add_to_dataset", default=None,
                        help="Dataset to add new results to")
    parser.add_argument("-e", "--use_existing_data", default=None,
                        help="Dataset to plot only (no new data generated)")
    parser.add_argument("-f", "--use_existing_fitter", action="store_true",
                        help="Flag to use existing fitter")
    parser.add_argument("-r", "--skip_risk_mpc", action="store_true",
                        help="Flag to skip risk based mpc runs")
    parser.add_argument("-s", "--skip_space_mpc", action="store_true",
                        help="Flag to skip space based mpc runs")
    parser.add_argument("-n", "--number", default=10, type=int, help="Number of repeats to run.")
    args = parser.parse_args()

    if args.use_existing_data is not None:
        # Load data
        with open(args.use_existing_data, "rb") as infile:
            data = pickle.load(infile)
    else:
        # Make data
        data = make_data(reuse_fitter=args.use_existing_fitter, skip_risk_mpc=args.skip_risk_mpc,
                         skip_space_mpc=args.skip_space_mpc, check_data=args.add_to_dataset,
                         nruns=args.number)

    if args.add_to_dataset is not None:
        with open(args.add_to_dataset, "rb") as infile:
            old_data = pickle.load(infile)

        add_keys = ['sim_no_control', 'sim_high', 'sim_split', 'sim_risk_opt', 'sim_space_opt',
                    'sim_risk_mpc', 'sim_space_mpc']
        for key in add_keys:
            old_data[key].extend(data[key])

        data = old_data

        with open(args.add_to_dataset, "wb") as outfile:
            pickle.dump(data, outfile)
    else:
        # Save data
        with open(os.path.join("Data", "data.pickle"), "wb") as outfile:
            pickle.dump(data, outfile)

    print("Dataset n = {0}".format(len(data['sim_no_control'])))

    make_plots(data, skip_risk_mpc=args.skip_risk_mpc, skip_space_mpc=args.skip_space_mpc)
