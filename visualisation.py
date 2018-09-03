"""Code to visualise simulation output."""

import simulation
import matplotlib.pyplot as plt
from matplotlib import patches, gridspec, animation
from matplotlib import colors as mcolors
import numpy as np

def err_bands(data, ax, xvals=None, col=None, alpha_range=None, lower_percentiles=None, label=None):
    """Generate plot showing point-wise percentile values."""

    if lower_percentiles is None:
        lower_percentiles = [0, 10, 20, 30, 40]
    if col is None:
        col = "Red"
    if alpha_range is None:
        alpha_range = [0.1, 0.5]

    upper_percentiles = [100 - lower for lower in lower_percentiles][::-1]
    percentile_data = []
    median_data = []

    if xvals is None:
        xvals = list(range(len(data)))

    if label is None:
        label = "Point-wise Median"

    for dat in data:
        median = [np.percentile(dat, 50)]
        lower = np.percentile(dat, lower_percentiles).tolist()
        upper = np.percentile(dat, upper_percentiles).tolist()

        percentile_data.append(lower + median + upper)
        median_data.append(median)

    percentile_data = np.array(percentile_data)

    alphas = np.linspace(alpha_range[0], alpha_range[1], len(lower_percentiles)).tolist()
    alphas = alphas + alphas[::-1]

    for idx in range(percentile_data.shape[1] - 1):
        ax.fill_between(xvals, percentile_data[:, idx], percentile_data[:, idx+1], color=col,
                        alpha=alphas[idx], linewidth=0, linestyle="None", edgecolor="none")

    ax.plot(xvals, median_data, color=col, linestyle="--", label=label)

def plot_node_network(nodes, dist_coupling, options=None, ax=None):
    """Plot network structure and node states from list of nodes.

    Options is a dictionary, following values are default:
        'min_node_radius': 0.1,
        'max_node_radius': 0.25,
        'population_vmin': None,
        'population_vmax': None,
        'scale_coupling_links': True,
        'coupling_link_min_size': 3,
        'coupling_link_max_size': 10,
        'vac_line_offset': 0.1,
        'vac_line_width': 3,
        'xlim': None,
        'ylim': None,
        'inf_cmap': "YlOrRd",
        'inf_vmin': 0.0,
        'inf_vmax': 1.0
    """

    positions = [node.position for node in nodes]
    population_sizes = np.array([
        np.sum([node.state[x] for x in simulation.HIGH_LIVE_STATES | simulation.LOW_LIVE_STATES])
        for node in nodes
    ])
    inf_props = np.array([
        (node.state[simulation.State.INF_H] + node.state[simulation.State.INF_L]) /
        population_sizes[i] for i, node in enumerate(nodes)
    ])
    vac_props = np.array([
        (node.state[simulation.State.VAC_H] + node.state[simulation.State.VAC_L]) /
        population_sizes[i] for i, node in enumerate(nodes)
    ])

    network_state = (positions, population_sizes, inf_props, vac_props)

    fig = _plot_network_state(network_state, dist_coupling, options=options, ax=ax)

    return fig

def plot_run_data(nodes, sim_run, dist_coupling, time=0, options=None, interact=False):
    """Plot network structure and state from simulation output, at given time.

    Options is a dictionary, following values are default:
        'show_regions': True,
        'min_node_radius': 0.1,
        'max_node_radius': 0.25,
        'population_vmin': None,
        'population_vmax': None,
        'scale_coupling_links': True,
        'coupling_link_min_size': 3,
        'coupling_link_max_size': 10,
        'vac_line_offset': 0.1,
        'vac_line_width': 3,
        'xlim': None,
        'ylim': None,
        'dpc_times': None,
    """

    if options is None:
        options = {}

    if options.get('dpc_max_time', None) is None:
        max_time = sim_run.run_data["Global"][-1][0]
    else:
        max_time = options['dpc_max_time']
    dpc_times = np.linspace(0, max_time, num=101)

    regions = sorted(list(set(node.region for node in nodes)))

    positions = [node.position for node in nodes]
    population_sizes, inf_props, vac_props = get_network_state(time, sim_run.run_data, nodes)
    dpc_data = _get_dpc_data([sim_run.run_data], dpc_times, nodes)

    network_state = (positions, population_sizes, inf_props, vac_props)

    # Make figure axes
    fig = plt.figure()
    fig.set_size_inches(8, 4.5)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    network_ax = fig.add_subplot(gs[0])

    height_ratios = [1, 0.05, 1] if options.get('show_regions', True) else [0, 0.05, 1]
    gs_dpc = gridspec.GridSpecFromSubplotSpec(3, 1, gs[1], height_ratios=height_ratios)
    gs_reg = gridspec.GridSpecFromSubplotSpec(1, 3*len(regions)-1, gs_dpc[0],
                                              width_ratios=list([1, 1, 0.5]*len(regions))[:-1])
    gs_glob = gridspec.GridSpecFromSubplotSpec(1, 2, gs_dpc[2])

    if options.get('show_regions', True):
        reg_axes = [fig.add_subplot(gs_reg[0])]
        reg_axes.append(fig.add_subplot(gs_reg[1]))
        reg_axes[0].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].yaxis.set_tick_params(labelleft=False, labelright=True)

        reg_ax = fig.add_subplot(gs_reg[0:2], frameon=False)
        reg_ax.set_title("Region " + str(regions[0]))
        reg_ax.set_xticks([])
        reg_ax.set_yticks([])

        for i, region in enumerate(regions[1:]):
            reg_axes.append(fig.add_subplot(gs_reg[3*i+3], sharey=reg_axes[0]))
            reg_axes.append(fig.add_subplot(gs_reg[3*i+4], sharey=reg_axes[1]))
            reg_axes[2*i+2].yaxis.set_tick_params(labelleft=True)
            reg_axes[2*i+2].xaxis.set_tick_params(labelbottom=False)
            reg_axes[2*i+3].yaxis.set_tick_params(labelleft=False, labelright=True)
            reg_axes[2*i+3].xaxis.set_tick_params(labelbottom=False)

            reg_ax = fig.add_subplot(gs_reg[3*i+3:3*i+5], frameon=False)
            reg_ax.set_title("Region " + str(region))
            reg_ax.set_xticks([])
            reg_ax.set_yticks([])
    else:
        reg_axes = None

    glob_axes = [fig.add_subplot(gs_glob[0])]
    glob_axes.append(fig.add_subplot(gs_glob[1]))
    glob_axes[0].set_title("High Risk")
    glob_axes[1].set_title("Low Risk")
    glob_ax = fig.add_subplot(gs_dpc[1], frameon=False)
    glob_ax.text(0.5, 0.5, 'Global', horizontalalignment='center', verticalalignment='center',
                 transform=glob_ax.transAxes, fontsize=24)
    glob_ax.set_xticks([])
    glob_ax.set_yticks([])


    if interact:
        options['timeline'] = time
        options['dpc_data'] = dpc_data
        network_fig_data = _plot_network_state(
            network_state, dist_coupling, ax=network_ax, options=options, for_animation=True)
        dpc_fig_data = _plot_dpc_data(dpc_data, glob_axes=glob_axes, reg_axes=reg_axes,
                                      regions=regions, options=options)
        fig_data = (fig, *network_fig_data, dpc_fig_data)
        interator = Interactor(sim_run.run_data, nodes, fig_data, options)
        plt.show()

    else:
        _plot_network_state(network_state, dist_coupling, ax=network_ax, options=options)
        _plot_dpc_data(
            dpc_data, regions=regions, glob_axes=glob_axes, reg_axes=reg_axes, options=options)
    return fig

def plot_dpc_data(nodes, sim_runs, options, nruns=1):
    """Plot DPC by region and risk structure.

    sim_runs can be a single simulation run, or a list of simulation runs.
    If sim_runs is a list of simulation runs, nruns specifies how many to show.
    """

    if isinstance(sim_runs, simulation.SimulationRun):
        runs = [sim_runs.run_data]
    else:
        runs = [sim.run_data for sim in sim_runs[:nruns]]

    regions = sorted(list(set(node.region for node in nodes)))

    if options.get('dpc_max_time', None) is None:
        max_time = np.max([run["Global"][-1][0] for run in runs])
    else:
        max_time = options['dpc_max_time']
    dpc_times = np.linspace(0, max_time, num=101)
    dpc_data = _get_dpc_data(runs, dpc_times, nodes)

    # Make figure axes
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)

    height_ratios = [1, 0.05, 1] if options.get('show_regions', True) else [0, 0.05, 1]
    gs_dpc = gridspec.GridSpecFromSubplotSpec(3, 1, gs[0], height_ratios=height_ratios)
    gs_reg = gridspec.GridSpecFromSubplotSpec(2, 3*len(regions)-1, gs_dpc[0],
                                              width_ratios=list([1, 1, 0.5]*len(regions))[:-1],
                                              height_ratios=[0.05, 1])
    gs_glob = gridspec.GridSpecFromSubplotSpec(1, 2, gs_dpc[2])

    if options.get('show_regions', True):
        reg_axes = [fig.add_subplot(gs_reg[1, 0])]
        reg_axes.append(fig.add_subplot(gs_reg[1, 1]))
        reg_axes[0].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].yaxis.set_tick_params(labelleft=False, labelright=True)
        reg_axes[0].set_title("High", fontsize=8)
        reg_axes[1].set_title("Low", fontsize=8)

        reg_ax = fig.add_subplot(gs_reg[0, 0:2], frameon=False)
        reg_ax.set_title("Region " + str(regions[0]))
        reg_ax.set_xticks([])
        reg_ax.set_yticks([])

        for i, region in enumerate(regions[1:]):
            reg_axes.append(fig.add_subplot(gs_reg[1, 3*i+3], sharey=reg_axes[0]))
            reg_axes.append(fig.add_subplot(gs_reg[1, 3*i+4], sharey=reg_axes[1]))
            reg_axes[2*i+2].yaxis.set_tick_params(labelleft=True)
            reg_axes[2*i+2].xaxis.set_tick_params(labelbottom=False)
            reg_axes[2*i+3].yaxis.set_tick_params(labelleft=False, labelright=True)
            reg_axes[2*i+3].xaxis.set_tick_params(labelbottom=False)
            reg_axes[2*i+2].set_title("High", fontsize=8)
            reg_axes[2*i+3].set_title("Low", fontsize=8)

            reg_ax = fig.add_subplot(gs_reg[0, 3*i+3:3*i+5], frameon=False)
            reg_ax.set_title("Region " + str(region))
            reg_ax.set_xticks([])
            reg_ax.set_yticks([])
    else:
        reg_axes = None

    glob_axes = [fig.add_subplot(gs_glob[0])]
    glob_axes.append(fig.add_subplot(gs_glob[1]))
    glob_axes[0].set_title("High Risk")
    glob_axes[1].set_title("Low Risk")
    glob_axes[0].set_xlabel("Time")
    glob_axes[1].set_xlabel("Time")
    glob_ax = fig.add_subplot(gs_dpc[1], frameon=False)
    glob_ax.text(0.5, 0.5, 'Global', horizontalalignment='center', verticalalignment='center',
                 transform=glob_ax.transAxes, fontsize=18)
    glob_ax.set_xticks([])
    glob_ax.set_yticks([])

    _plot_dpc_data(dpc_data, glob_axes=glob_axes, reg_axes=reg_axes, options=options,
                   regions=regions)

    fig.tight_layout()

    if options.get("show_regions", True):
        return fig, reg_axes, glob_axes

    return fig, None, glob_axes

def animate_run_data(nodes, run_data, dist_coupling, video_length=10, options=None, save_file=None):
    """Show/save animation of network epidemic simulation."""

    default_options = {
        'show_dpcs': False,
        'dpc_data': None,
        'dpc_max_time': None,
        'regions': None,
        'min_node_radius': 0.1,
        'max_node_radius': 0.25,
        'population_vmin': 10,
        'population_vmax': 200,
        'scale_coupling_links': True,
        'coupling_link_min_size': 3,
        'coupling_link_max_size': 10,
        'vac_line_offset': 0.1,
        'vac_line_width': 3,
        'xlim': None,
        'ylim': None
    }

    if options is None:
        options = {}

    for key, val in default_options.items():
        if key not in options:
            options[key] = val

    fps = 30
    nframes = fps * video_length

    regions = sorted(list(set(node.region for node in nodes)))
    if options.get('dpc_max_time', None) is None:
        max_time = run_data["Global"][-1][0]
    else:
        max_time = options['dpc_max_time']
    dpc_times = np.linspace(0, max_time, num=101)
    dpc_data = _get_dpc_data([run_data], dpc_times, nodes)

    population_sizes, inf_props, vac_props = get_network_state(0, run_data, nodes)
    positions = [node.position for node in nodes]

    network_state = (positions, population_sizes, inf_props, vac_props)
    options['timeline'] = 0

    all_data = []
    anim_times = np.linspace(0, max_time, num=nframes)

    for anim_t in anim_times:
        pop_sizes, inf_props, vac_props = get_network_state(anim_t, run_data, nodes)
        node_radii = scale_values(normalise_values(
            np.sqrt(pop_sizes), vmin=np.sqrt(options['population_vmin']),
            vmax=np.sqrt(options['population_vmax'])), vmin=2*options['min_node_radius'],
                                  vmax=2*options['max_node_radius'])
        all_data.append(list(zip(node_radii, inf_props, vac_props)))

    # Make figure axes
    fig = plt.figure()
    fig.set_size_inches(8, 4.5)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    network_ax = fig.add_subplot(gs[0])

    height_ratios = [1, 0.05, 1] if options.get('show_regions', True) else [0, 0.05, 1]
    gs_dpc = gridspec.GridSpecFromSubplotSpec(3, 1, gs[1], height_ratios=height_ratios)
    gs_reg = gridspec.GridSpecFromSubplotSpec(1, 3*len(regions)-1, gs_dpc[0],
                                              width_ratios=list([1, 1, 0.5]*len(regions))[:-1])
    gs_glob = gridspec.GridSpecFromSubplotSpec(1, 2, gs_dpc[2])

    if options.get('show_regions', True):
        reg_axes = [fig.add_subplot(gs_reg[0])]
        reg_axes.append(fig.add_subplot(gs_reg[1]))
        reg_axes[0].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].xaxis.set_tick_params(labelbottom=False)
        reg_axes[1].yaxis.set_tick_params(labelleft=False, labelright=True)

        reg_ax = fig.add_subplot(gs_reg[0:2], frameon=False)
        reg_ax.set_title("Region " + str(regions[0]))
        reg_ax.set_xticks([])
        reg_ax.set_yticks([])

        for i, region in enumerate(regions[1:]):
            reg_axes.append(fig.add_subplot(gs_reg[3*i+3], sharey=reg_axes[0]))
            reg_axes.append(fig.add_subplot(gs_reg[3*i+4], sharey=reg_axes[1]))
            reg_axes[2*i+2].yaxis.set_tick_params(labelleft=True)
            reg_axes[2*i+2].xaxis.set_tick_params(labelbottom=False)
            reg_axes[2*i+3].yaxis.set_tick_params(labelleft=False, labelright=True)
            reg_axes[2*i+3].xaxis.set_tick_params(labelbottom=False)

            reg_ax = fig.add_subplot(gs_reg[3*i+3:3*i+5], frameon=False)
            reg_ax.set_title("Region " + str(region))
            reg_ax.set_xticks([])
            reg_ax.set_yticks([])
    else:
        reg_axes = None

    glob_axes = [fig.add_subplot(gs_glob[0])]
    glob_axes.append(fig.add_subplot(gs_glob[1]))
    glob_axes[0].set_title("High Risk")
    glob_axes[1].set_title("Low Risk")
    glob_ax = fig.add_subplot(gs_dpc[1], frameon=False)
    glob_ax.text(0.5, 0.5, 'Global', horizontalalignment='center', verticalalignment='center',
                 transform=glob_ax.transAxes, fontsize=24)
    glob_ax.set_xticks([])
    glob_ax.set_yticks([])

    node_patches, vac_patches = _plot_network_state(
        network_state, dist_coupling, ax=network_ax, options=options, for_animation=True)
    time_lines = _plot_dpc_data(dpc_data, glob_axes=glob_axes, reg_axes=reg_axes,
                                options=options, regions=regions)

    gs.tight_layout(fig)

    for node_p, vac_p, time_line_p in zip(node_patches, vac_patches, time_lines):
        node_p.set_animated(True)
        vac_p.set_animated(True)
        time_line_p.set_animated(True)

    def update(frame_number):
        update_node_patches((node_patches, vac_patches), all_data[frame_number], options)
        for line in time_lines:
            line.set_xdata(anim_times[frame_number])

        return node_patches + vac_patches + time_lines

    im_ani = animation.FuncAnimation(fig, update, interval=1000*video_length/nframes,
                                     frames=nframes, blit=True, repeat=True)

    if save_file is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, codec="h264")
        im_ani.save(save_file+'.mp4', writer=writer, dpi=200)

    return im_ani

def plot_control(sim_run, end_time, risk_based=True, comparison=None, comparison_args=None, ax=None,
                 regions=None, **stackplot_args):
    """Plot control scheme from simulation run.

    Arguments:
        risk_based      - whether control is risk based, otherwise spatial. Default True
        comparison      - Control as function of time to plot as comparison. Default None
        comparison_args - Dictionary of additional kwargs to pass to comparison plot. Default None
        ax              - Axis to plot onto. If None creates new figure. Default None
        regions         - List of region names if not risk based. Default None
        stackplot_args  - Additional kwargs to pass to stackplot.
    """

    if risk_based:
        update_times = [x[0] for x in sim_run.control]
        end_times = update_times[1:] + [end_time]
        times = []
        high = []
        low = []

        for i, (start, end) in enumerate(zip(update_times, end_times)):
            new_times = np.linspace(start, end, 51, endpoint=False)
            times.extend(new_times)
            high.extend([np.sum(sim_run.control[i][1].control(t)[0::2]) for t in new_times])
            low.extend([np.sum(sim_run.control[i][1].control(t)[1::2]) for t in new_times])

        default = {
            "labels": ["High", "Low"],
            "colors": ["firebrick", "skyblue"],
        }
        for key, val in default.items():
            if key not in stackplot_args.keys():
                stackplot_args[key] = val

        if ax is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
        else:
            ax1 = ax

        ax1.stackplot(times, high, low, **stackplot_args)
        if comparison is not None:
            comparison_times = np.linspace(0, end_time, 101)
            if comparison_args is None:
                comparison_args = {}
            ax1.plot(comparison_times, [np.sum(comparison(t)[0::2]) for t in comparison_times],
                     "k--", **comparison_args)

        ax1.set_xlim([-0.1, end_time + 0.1])
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Control Proportion")

        if ax is None:
            fig.tight_layout()

        return ax1

    else:
        update_times = [x[0] for x in sim_run.control]
        end_times = update_times[1:] + [end_time]
        times = []


        if regions is None:
            regions = [str(x) for x in range(1, 1 + int(len(sim_run.control[0][1].control(0))/2))]
        
        # Order controls by risk first, then region
        ordering = [(2*x)%6 + (2*x)//6 for x in range(2*len(regions))]

        all_controls = [[] for _ in range(len(regions)*2)]

        for i, (start, end) in enumerate(zip(update_times, end_times)):
            new_times = np.linspace(start, end, 51, endpoint=False)
            times.extend(new_times)
            for j, control in enumerate(all_controls):
                control.extend([np.sum(sim_run.control[i][1].control(t)[ordering[j]])
                                for t in new_times])

        default = {
            "labels": [region + " " + risk for risk in ["High", "Low"] for region in regions],
            "colors": [mcolors.to_rgba("C{}".format(i), alpha=alpha) for alpha in [1.0, 0.5]
                       for i, _ in enumerate(regions)],
        }
        for key, val in default.items():
            if key not in stackplot_args.keys():
                stackplot_args[key] = val

        if ax is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
        else:
            ax1 = ax

        ax1.stackplot(times, *all_controls, **stackplot_args)
        if comparison is not None:
            if comparison_args is None:
                comparison_args = {}
            ax1.plot(times, [comparison(t)[0] for t in times], "k--", **comparison_args)

        ax1.set_xlim([-0.1, end_time + 0.1])
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Control Proportion")

        if ax is None:
            fig.tight_layout()

        return ax1

def _plot_network_state(network_state, dist_coupling, ax=None, options=None, for_animation=False):
    """Plot network structure."""

    if options is None:
        options = {}

    default_options = {
        'min_node_radius': 0.1,
        'max_node_radius': 0.25,
        'population_vmin': 0,
        'population_vmax': 100,
        'scale_coupling_links': True,
        'coupling_link_min_size': 3,
        'coupling_link_max_size': 10,
        'node_alpha': 0.75,
        'vac_line_offset': 0.1,
        'vac_line_width': 3,
        'xlim': None,
        'ylim': None,
        'inf_cmap': plt.get_cmap("YlOrRd"),
        'inf_vmin': 0.0,
        'inf_vmax': 1.0
    }

    if ax is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    for key, val in default_options.items():
        if key not in options:
            options[key] = val

    positions, population_sizes, inf_props, vac_props = network_state
    n_nodes = len(population_sizes)

    coupling_mask = np.ones(np.array(dist_coupling).shape, dtype=bool)
    np.fill_diagonal(coupling_mask, 0)

    normed_inf = normalise_values(inf_props, vmin=options['inf_vmin'], vmax=options['inf_vmax'])

    normed_coupling = scale_values(normalise_values(dist_coupling,
                                                    vmin=np.min(dist_coupling[coupling_mask]),
                                                    vmax=np.max(dist_coupling[coupling_mask])),
                                   vmin=options['coupling_link_min_size'],
                                   vmax=options['coupling_link_max_size'])

    node_sizes = scale_values(normalise_values(np.sqrt(population_sizes),
                                               vmin=np.sqrt(options['population_vmin']),
                                               vmax=np.sqrt(options['population_vmax'])),
                              vmin=2*options['min_node_radius'], vmax=2*options['max_node_radius'])

    ax1.set_xticks([])
    ax1.set_yticks([])
    if options['xlim'] is not None:
        ax1.set_xlim(options['xlim'])
    else:
        xlim = [min([x[0] for x in positions]) - options['max_node_radius'] -
                2*options['vac_line_offset'], max([x[0] for x in positions]) +
                options['max_node_radius'] + 2*options['vac_line_offset']]
        ax1.set_xlim(xlim)
    if options['ylim'] is not None:
        ax1.set_xlim(options['ylim'])
    else:
        ylim = [min([x[1] for x in positions]) - options['max_node_radius'] -
                2*options['vac_line_offset'], max([x[1] for x in positions]) +
                options['max_node_radius'] + 2*options['vac_line_offset']]
        ax1.set_ylim(ylim)
    ax1.set_aspect("equal")

    zipped_data = zip(positions, node_sizes, normed_inf, vac_props)
    node_patches, vac_patches = create_node_patches(zipped_data, options)
    for node_patch, vac_patch in zip(node_patches, vac_patches):
        ax1.add_patch(node_patch)
        ax1.add_patch(vac_patch)

    for i in range(n_nodes):
        for j in range(i):
            if dist_coupling[i, j] > 0:
                if options['scale_coupling_links']:
                    lw = normed_coupling[i, j]
                else:
                    lw = (options['coupling_link_min_size'] + options['coupling_link_max_size'])/2
                ax1.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]],
                         '-', color="lightgray", lw=lw, zorder=0)

    if ax is None:
        fig.tight_layout()

    if for_animation:
        return node_patches, vac_patches
    return ax1.figure

def _plot_dpc_data(dpc_data, glob_axes=None, reg_axes=None, regions=None, options=None):

    if options is None:
        options = {}

    if options.get('timeline', None) is not None:
        time_lines = []
    else:
        time_lines = None

    if regions is None:
        regions = []

    if reg_axes is not None:
        for i, region in enumerate(regions):

            reg_axes[2*i].plot(dpc_data['times'], dpc_data['Region'+str(region)][0], color='green',
                               alpha=options.get('alpha', 1.0))
            reg_axes[2*i].plot(dpc_data['times'], dpc_data['Region'+str(region)][1], color='red',
                               alpha=options.get('alpha', 1.0))

            reg_axes[2*i+1].plot(dpc_data['times'], dpc_data['Region'+str(region)][3],
                                 color='green', alpha=options.get('alpha', 1.0))
            reg_axes[2*i+1].plot(dpc_data['times'], dpc_data['Region'+str(region)][4], color='red',
                                 alpha=options.get('alpha', 1.0))

            if options.get("show_vac", True):
                reg_axes[2*i].plot(dpc_data['times'], dpc_data['Region'+str(region)][2],
                                   color='purple', alpha=options.get('alpha', 1.0))
                reg_axes[2*i+1].plot(dpc_data['times'], dpc_data['Region'+str(region)][5],
                                     color='purple', alpha=options.get('alpha', 1.0))

    if glob_axes is not None:
        glob_axes[0].plot(dpc_data['times'], dpc_data['Global'][0], color='green',
                        alpha=options.get('alpha', 1.0))
        glob_axes[0].plot(dpc_data['times'], dpc_data['Global'][1], color='red',
                        alpha=options.get('alpha', 1.0))

        glob_axes[1].plot(dpc_data['times'], dpc_data['Global'][3], color='green',
                        alpha=options.get('alpha', 1.0))
        glob_axes[1].plot(dpc_data['times'], dpc_data['Global'][4], color='red',
                        alpha=options.get('alpha', 1.0))

        if options.get("show_vac", True):
            glob_axes[0].plot(dpc_data['times'], dpc_data['Global'][2], color='purple',
                            alpha=options.get('alpha', 1.0))
            glob_axes[1].plot(dpc_data['times'], dpc_data['Global'][5], color='purple',
                            alpha=options.get('alpha', 1.0))

    if time_lines is not None:
        for ax in glob_axes:
            line = ax.axvline(options['timeline'], ls='--', color="darkgrey")
            time_lines.append(line)
        for ax in reg_axes:
            line = ax.axvline(options['timeline'], ls='--', color="darkgrey", linewidth=0.5)
            time_lines.append(line)

    return time_lines

def scale_values(values, vmin=0, vmax=1):
    """Scale normalised values to between vmin and vmax"""

    if np.any(np.ma.fix_invalid(values) < 0) or np.any(np.ma.fix_invalid(values) > 1):
        raise ValueError("Values not between 0 and 1!")

    return vmin + np.array(values)*(vmax - vmin)

def normalise_values(values, vmin=None, vmax=None):
    """Generate normalised values between 0 and 1."""

    if vmin is None:
        vmin = np.min(np.ma.fix_invalid(values))
    if vmax is None:
        vmax = np.max(np.ma.fix_invalid(values))

    if vmin == vmax:
        return np.full_like(values, vmin)

    return (np.clip(values, vmin, vmax) - vmin)/(vmax - vmin)

def create_node_patches(zipped_data, options):
    """Generate list of patches to plot.

    zipped_data contains: zip(positions, radii, inf_props, vac_props)
    options is dictionary with following keys:
        'inf_cmap':         cmap to use for infection proportions (optional)
        'node_alpha':       alpha channel for node patches
        'vac_line_offset':  absolute offset from node edge to position vaccinated line
        'vac_line_width':   width for vaccination line
    """

    node_patches = []
    vac_patches = []

    if options.get("inf_cmap", None) is None:
        cmap = plt.get_cmap("YlOrRd")
    else:
        cmap = options['inf_cmap']
    cmap.set_under("black")

    for pos, radius, inf, vac in zipped_data:
        with np.errstate(invalid="ignore"):
            node_col = cmap(inf)
        node_patches.append(patches.Ellipse(pos, radius, radius, color=node_col, zorder=5,
                                            alpha=options['node_alpha']))
        if vac == 1:
            theta1 = None
            theta2 = None
        else:
            theta1 = 450-vac*360
            theta2 = 90
        vac_patches.append(patches.Arc(pos, radius+options['vac_line_offset'],
                                       radius+options['vac_line_offset'], theta1=theta1,
                                       theta2=theta2, facecolor=None,
                                       linewidth=options['vac_line_width'], edgecolor="Purple",
                                       fill=False, zorder=5))

    return node_patches, vac_patches

def update_node_patches(patch_data, zipped_data, options):
    node_patches, vac_patches = patch_data

    cmap = plt.get_cmap("autumn_r")
    cmap.set_under("black")

    for i, (radius, inf, vac) in enumerate(zipped_data):
        with np.errstate(invalid="ignore"):
            node_col = cmap(inf)
        node_patches[i].width = radius
        node_patches[i].height = radius
        node_patches[i].set_fc(node_col)
        node_patches[i].set_ec(node_col)

        vac_patches[i].width = radius+options['vac_line_offset']
        vac_patches[i].height = radius+options['vac_line_offset']
        vac_patches[i].theta1 = 450-vac*360

def get_network_state(time, run_data, nodes):
    n_nodes = len(nodes)
    population_sizes = np.zeros(n_nodes)
    inf_props = np.zeros(n_nodes)
    vac_props = np.zeros(n_nodes)

    for node in nodes:
        times = [x[0] for x in run_data["Node"+str(node.id)]]
        idx = np.searchsorted(times, time, side="right")
        state = run_data["Node"+str(node.id)][idx-1][1:]

        population_sizes[node.id] = np.sum([
            state[x] for x in simulation.HIGH_LIVE_STATES | simulation.LOW_LIVE_STATES])

        if population_sizes[node.id] > 0:
            inf_props[node.id] = ((state[simulation.State.INF_H] + state[simulation.State.INF_L]) /
                                  population_sizes[node.id])

            vac_props[node.id] = ((state[simulation.State.VAC_H] + state[simulation.State.VAC_L]) /
                                  population_sizes[node.id])
        else:
            inf_props[node.id] = np.nan
            vac_props[node.id] = 0

    return (population_sizes, inf_props, vac_props)

def _get_dpc_data(run_data, times, nodes):
    """Extract DPC data from list of simulation run_data objects."""

    regions = sorted(list(set(node.region for node in nodes)))

    dpc_data = {"Region"+str(x): [np.zeros((len(times), len(run_data))) for _ in range(6)]
                for x in regions}
    dpc_data['times'] = times
    dpc_data['Global'] = [np.zeros((len(times), len(run_data))) for _ in range(6)]

    for run_num, run in enumerate(run_data):
        for i, dpc_t in enumerate(dpc_data['times']):
            event_times = [x[0] for x in run["Global"]]
            idx = np.searchsorted(event_times, dpc_t, side="right")
            state = run["Global"][idx-1][1:]

            sus_h_num = state[simulation.State.SUS_H]
            inf_h_num = state[simulation.State.INF_H]
            vac_h_num = state[simulation.State.VAC_H]
            sus_l_num = state[simulation.State.SUS_L]
            inf_l_num = state[simulation.State.INF_L]
            vac_l_num = state[simulation.State.VAC_L]

            dpc_data["Global"][0][i, run_num] += sus_h_num
            dpc_data["Global"][1][i, run_num] += inf_h_num
            dpc_data["Global"][2][i, run_num] += vac_h_num
            dpc_data["Global"][3][i, run_num] += sus_l_num
            dpc_data["Global"][4][i, run_num] += inf_l_num
            dpc_data["Global"][5][i, run_num] += vac_l_num

            for region in regions:
                event_times = [x[0] for x in run["Region" + str(region)]]
                idx = np.searchsorted(event_times, dpc_t, side="right")
                state = run["Region" + str(region)][idx-1][1:]

                sus_h_num = state[simulation.State.SUS_H]
                inf_h_num = state[simulation.State.INF_H]
                vac_h_num = state[simulation.State.VAC_H]
                sus_l_num = state[simulation.State.SUS_L]
                inf_l_num = state[simulation.State.INF_L]
                vac_l_num = state[simulation.State.VAC_L]

                dpc_data["Region" + str(region)][0][i, run_num] += sus_h_num
                dpc_data["Region" + str(region)][1][i, run_num] += inf_h_num
                dpc_data["Region" + str(region)][2][i, run_num] += vac_h_num
                dpc_data["Region" + str(region)][3][i, run_num] += sus_l_num
                dpc_data["Region" + str(region)][4][i, run_num] += inf_l_num
                dpc_data["Region" + str(region)][5][i, run_num] += vac_l_num

    return dpc_data

class Interactor:
    def __init__(self, run_data, nodes, fig_data, options):
        self.options = options

        self.run_data = run_data
        self.nodes = nodes
        self.fig, self.node_patches, self.vac_patches, self.time_lines = fig_data

        # self.n_nodes = len(node_patches)

        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.press = None
        self.background = None

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes not in [x.axes for x in self.time_lines]:
            return

        x0 = self.time_lines[-1].get_xdata()
        if isinstance(x0, list):
            x0 = x0[0]
        self.press = x0, event.xdata

        # draw everything but the time lines and store the pixel buffer
        self.background = []
        canvas = self.fig.canvas
        for line in self.time_lines:
            line.set_animated(True)
        canvas.draw()
        for line in self.time_lines:
            axes = line.axes
            self.background.append(canvas.copy_from_bbox(line.axes.bbox))

            # now redraw just the rectangle
            axes.draw_artist(line)

            # and blit just the redrawn area
            canvas.blit(axes.bbox)
        
        print(event)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None:
            return
        if event.inaxes not in [x.axes for x in self.time_lines]:
            return
        x0, xpress = self.press
        time = np.clip(x0 + event.xdata - xpress, 0, max(self.options['dpc_data']['times']))
        for line in self.time_lines:
            line.set_xdata(time)

        canvas = self.fig.canvas
        for line, background in zip(self.time_lines, self.background):
            axes = line.axes
            # restore the background region
            canvas.restore_region(background)

            # redraw just the current rectangle
            axes.draw_artist(line)

            # blit just the redrawn area
            canvas.blit(axes.bbox)
        
        print(event)

    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None:
            return
        if event.inaxes not in [x.axes for x in self.time_lines]:
            return
        x0, xpress = self.press
        time = np.clip(x0 + event.xdata - xpress, 0, max(self.options['dpc_data']['times']))
        pop_sizes, inf_props, vac_props = get_network_state(time, self.run_data, self.nodes)
        node_sizes = scale_values(normalise_values(
            np.sqrt(pop_sizes), vmin=np.sqrt(self.options['population_vmin']),
            vmax=np.sqrt(self.options['population_vmax'])), vmin=2*self.options['min_node_radius'],
                                  vmax=2*self.options['max_node_radius'])
        update_node_patches(
            (self.node_patches, self.vac_patches), zip(node_sizes, inf_props, vac_props),
            self.options)
        self.press = None
        for line in self.time_lines:
            line.set_animated(False)
        self.background = None
        self.fig.canvas.draw()

        print(event)