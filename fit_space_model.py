"""Code for parameter and function fitting of space structured OCT model to simulation data."""

import copy
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
import simulation
import space_model
import control_tools
import fitting
import visualisation

class SpaceFitter:
    """Class for fitting space structured OCT model, handling likelihood and linear regression."""

    def __init__(self, parent):
        self.data = {}
        self.linear_fits = {}
        self.parent_fitter = parent

    def __repr__(self):
        repr_str = "\nSpace Model Fitter\n" + "-"*20 + "\n\n"

        if self.linear_fits:
            repr_str += repr(self.linear_fits['likelihood'])
        else:
            repr_str += "No linear fits to show\n" + "-"*20 + "\n\n"

        return repr_str

    def fit(self, bounds, start):
        """Fit space model to generated data."""

        self.linear_fits['likelihood'] = SpaceFitterLikelihood(self)
        self.linear_fits['likelihood'].fit(bounds, start)

    def assess(self, fitter, save_folder=None, control=None, likelihood_map=False,
               initial_nodes=None, max_control_rate=100):
        """Assess fit quality."""

        if "init_conds" not in self.data:
            self.optimise(fitter, initial_nodes)

        print("Assessing {0} space model...".format(fitter.name))

        # Generate model run data
        if control is None:
            controller = space_model.no_control_policy
        else:
            controller = control

        state_init = self.data['init_conds']

        model_params = {
            'birth_rate': self.parent_fitter.sim_params['birth_rate'],
            'death_rate': self.parent_fitter.sim_params['death_rate'],
            'removal_rate': self.parent_fitter.sim_params['removal_rate'],
            'recov_rate': self.parent_fitter.sim_params['recov_rate'],
            'state_init': state_init,
            'times': self.parent_fitter.data['dpc_times'],
            'max_control_rate': max_control_rate,
            'high_alloc_cost': 0,
            'low_alloc_cost': 0
        }
        model = space_model.SpaceModel(model_params, fitter)
        model_run = model.run_policy(controller)
        run_data = np.array([model_run.state(t) for t in model_params['times']])

        self._assess_dpc(run_data, control=control, save_folder=save_folder)

        self._calc_metrics(fitter, run_data, control=control)

        if likelihood_map:
            self._likelihood_map(fitter, save_folder=save_folder)

    def optimise(self, fitter, initial_nodes=None, max_control_rate=100, run_sims=True):
        """Optimise control on given fit."""

        print("Optimising space based control...")

        if initial_nodes is None:
            nodes_init = fitting.randomise_infection(self.parent_fitter.nodes, nrand=5)
        else:
            nodes_init = copy.deepcopy(initial_nodes)

        state_init = np.zeros(18)
        for node in nodes_init:
            if node.region == "A":
                region = 0
            elif node.region == "B":
                region = 1
            elif node.region == "C":
                region = 2

            states = sorted(simulation.HIGH_LIVE_STATES | simulation.LOW_LIVE_STATES)
            for i, state_val in enumerate(states):
                state_init[6*region+i] += node.state[state_val]

        model_params = {
            'birth_rate': self.parent_fitter.sim_params['birth_rate'],
            'death_rate': self.parent_fitter.sim_params['death_rate'],
            'removal_rate': self.parent_fitter.sim_params['removal_rate'],
            'recov_rate': self.parent_fitter.sim_params['recov_rate'],
            'state_init': state_init,
            'times': self.parent_fitter.data['dpc_times'],
            'max_control_rate': max_control_rate,
            'high_alloc_cost': 0,
            'low_alloc_cost': 0
        }
        model = space_model.SpaceModel(model_params, fitter)
        bocop_run = model.run_bocop(verbose=False, init_policy=space_model.even_control_policy)
        opt_control = bocop_run.control
        self.data['opt_control'] = opt_control
        self.data['init_conds'] = state_init
        self.data['dpc_times'] = model_params['times']

        if run_sims:
            self._get_sim_dpcs(model, opt_control, initial_nodes)

    def _get_sim_dpcs(self, model, opt_control, initial_nodes=None):
        """Run simulations with and without control for assessment DPCs."""

        sim_params = copy.deepcopy(self.parent_fitter.sim_params)

        # Run no control sims
        print("Generating space testing set...")

        stored_data = {}
        input_events = []
        for _ in range(10):
            if initial_nodes is None:
                region_init = {
                    "A": (self.data['init_conds'][1], self.data['init_conds'][4]),
                    "B": (self.data['init_conds'][7], self.data['init_conds'][10]),
                    "C": (self.data['init_conds'][13], self.data['init_conds'][16])
                }
                nodes_init = fitting.randomise_init_infs(self.parent_fitter.nodes, region_init)
            else:
                nodes_init = copy.deepcopy(initial_nodes)

            simulator = simulation.Simulator(nodes_init, self.parent_fitter.risk_coupling,
                                             self.parent_fitter.dist_coupling, sim_params,
                                             controller=None)
            all_runs = simulator.run_simulation(nruns=10, verbose=False)
            self.parent_fitter._store_dpc_data(all_runs, data_store=stored_data, include_vac=False)
            input_events += self.parent_fitter._extract_event_data(all_runs)

        for key, val in stored_data.items():
            self.data[key] = val
        self.data["assessment_events"] = input_events

        sim_params['update_control_on_all_events'] = True
        sim_params['vac_rate'] = 1.0
        sim_params["controller_args"] = {
            "oc_model": model,
            "policy": opt_control
        }

        # Run simulations with control
        stored_data = {}
        input_events = []
        for _ in range(10):
            if initial_nodes is None:
                region_init = {
                    "A": (self.data['init_conds'][1], self.data['init_conds'][4]),
                    "B": (self.data['init_conds'][7], self.data['init_conds'][10]),
                    "C": (self.data['init_conds'][13], self.data['init_conds'][16])
                }
                nodes_init = fitting.randomise_init_infs(self.parent_fitter.nodes, region_init)
            else:
                nodes_init = copy.deepcopy(initial_nodes)

            simulator = simulation.Simulator(nodes_init, self.parent_fitter.risk_coupling,
                                             self.parent_fitter.dist_coupling, sim_params,
                                             controller=control_tools.SpacePolicyController)
            all_runs = simulator.run_simulation(nruns=10, verbose=False)
            self.parent_fitter._store_dpc_data(all_runs, data_store=stored_data, include_vac=True)
            input_events += self.parent_fitter._extract_event_data(all_runs)

        for key, val in stored_data.items():
            self.data["controlled_" + key] = val
        self.data["controlled_assessment_events"] = input_events

    def _assess_dpc(self, run_data, control=None, save_folder=None, save_name="Linear_Likelihood"):
        """Plot DPC data fit against simulation data."""

        data = self.data

        fig = plt.figure()

        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1])
        gs_reg = gridspec.GridSpecFromSubplotSpec(1, 8, gs[0])
        gs_glob = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2])

        reg_axes = []
        glob_axes = []

        for reg in range(3):
            for risk in range(2):
                reg_axes.append(fig.add_subplot(gs_reg[3*reg+risk]))
                if control is None:
                    reg_axes[2*reg+risk].plot(data['dpc_times'], data['dpc_sus'][reg, risk, :, :],
                                              'g--', alpha=0.1)
                    reg_axes[2*reg+risk].plot(data['dpc_times'], data['dpc_inf'][reg, risk, :, :],
                                              'r--', alpha=0.1)

                else:
                    reg_axes[2*reg+risk].plot(
                        data['dpc_times'], data['controlled_dpc_sus'][reg, risk, :, :],
                        'g--', alpha=0.1)
                    reg_axes[2*reg+risk].plot(
                        data['dpc_times'], data['controlled_dpc_inf'][reg, risk, :, :],
                        'r--', alpha=0.1)
                    reg_axes[2*reg+risk].plot(
                        data['dpc_times'], data['controlled_dpc_vac'][reg, risk, :, :],
                        '--', color="purple", alpha=0.1)

                reg_axes[2*reg+risk].plot(data['dpc_times'], run_data[:, 6*reg+3*risk], 'g-', lw=2)
                reg_axes[2*reg+risk].plot(data['dpc_times'], run_data[:, 1+6*reg+3*risk], 'r-',
                                          lw=2)

                if control is not None:
                    reg_axes[2*reg+risk].plot(data['dpc_times'], run_data[:, 2+6*reg+3*risk], '-',
                                              color='purple', lw=2)

        glob_ax = fig.add_subplot(gs[1], frameon=False)
        glob_ax.text(0.5, 0.5, 'Global', horizontalalignment='center', verticalalignment='center',
                     transform=glob_ax.transAxes, fontsize=24)
        glob_ax.set_xticks([])
        glob_ax.set_yticks([])

        for risk in range(2):
            glob_axes.append(fig.add_subplot(gs_glob[risk]))
            if control is None:
                glob_axes[risk].plot(data['dpc_times'],
                                     np.sum(data['dpc_sus'][:, risk, :, :], axis=0), 'g--',
                                     alpha=0.1)
                glob_axes[risk].plot(data['dpc_times'],
                                     np.sum(data['dpc_inf'][:, risk, :, :], axis=0), 'r--',
                                     alpha=0.1)

            else:
                glob_axes[risk].plot(data['dpc_times'],
                                     np.sum(data['controlled_dpc_sus'][:, risk, :, :], axis=0),
                                     'g--', alpha=0.1)
                glob_axes[risk].plot(data['dpc_times'],
                                     np.sum(data['controlled_dpc_inf'][:, risk, :, :], axis=0),
                                     'r--', alpha=0.1)
                glob_axes[risk].plot(data['dpc_times'],
                                     np.sum(data['controlled_dpc_vac'][:, risk, :, :], axis=0),
                                     '--', color="purple", alpha=0.1)

            glob_axes[risk].plot(data['dpc_times'], np.sum(run_data[:, 3*risk::6], axis=1), 'g-',
                                 lw=2)
            glob_axes[risk].plot(data['dpc_times'], np.sum(run_data[:, 1+3*risk::6], axis=1), 'r-',
                                 lw=2)

            if control is not None:
                glob_axes[risk].plot(data['dpc_times'], np.sum(run_data[:, 2+3*risk::6], axis=1),
                                     '-', color="purple", lw=2)

        if control is None:
            fig_name = save_name + "_DPCQuality.pdf"
        else:
            fig_name = save_name + "_Controlled_DPCQuality.pdf"

        fig.tight_layout()
        if save_folder is None:
            fig.savefig(fig_name)
        else:
            fig.savefig(os.path.join(save_folder, fig_name))
        plt.close(fig)

        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1])
        gs_reg = gridspec.GridSpecFromSubplotSpec(1, 8, gs[0])
        gs_glob = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2])

        reg_axes = []
        glob_axes = []

        for reg in range(3):
            for risk in range(2):
                reg_axes.append(fig.add_subplot(gs_reg[3*reg+risk]))
                if control is None:
                    visualisation.err_bands(
                        data['dpc_inf'][reg, risk, :, :], reg_axes[2*reg+risk],
                        data['dpc_times'], col="red", alpha_range=[0.05, 0.35],
                        lower_percentiles=np.linspace(0, 50, 21, endpoint=False))
                else:
                    visualisation.err_bands(
                        self.data['controlled_dpc_inf'][reg, risk, :, :], reg_axes[2*reg+risk],
                        data['dpc_times'], col="red", alpha_range=[0.05, 0.35],
                        lower_percentiles=np.linspace(0, 50, 21, endpoint=False))

                reg_axes[2*reg+risk].plot(data['dpc_times'], run_data[:, 1+6*reg+3*risk], 'r-',
                                          lw=2)

        glob_ax = fig.add_subplot(gs[1], frameon=False)
        glob_ax.text(0.5, 0.5, 'Global', horizontalalignment='center',
                     verticalalignment='center', transform=glob_ax.transAxes, fontsize=24)
        glob_ax.set_xticks([])
        glob_ax.set_yticks([])

        for risk in range(2):
            glob_axes.append(fig.add_subplot(gs_glob[risk]))
            if control is None:
                visualisation.err_bands(np.sum(data['dpc_inf'][:, risk, :, :], axis=0),
                                        glob_axes[risk], data['dpc_times'], col="red",
                                        alpha_range=[0.05, 0.35],
                                        lower_percentiles=np.linspace(0, 50, 21, endpoint=False))
            else:
                visualisation.err_bands(
                    np.sum(self.data['controlled_dpc_inf'][:, risk, :, :], axis=0),
                    glob_axes[risk], data['dpc_times'], col="red", alpha_range=[0.05, 0.35],
                    lower_percentiles=np.linspace(0, 50, 21, endpoint=False))

            glob_axes[risk].plot(data['dpc_times'], np.sum(run_data[:, 1+3*risk::6], axis=1),
                                 'r-', lw=2)

        if control is None:
            fig_name = save_name + "_Median.pdf"
        else:
            fig_name = save_name + "_Controlled_Median.pdf"

        fig.tight_layout()
        if save_folder is None:
            fig.savefig(fig_name)
        else:
            fig.savefig(os.path.join(save_folder, fig_name))
        plt.close(fig)

    def _calc_metrics(self, fitter, run_data, control=None):
        """Calculate model selection metrics."""

        # Calculate SSE
        if control is None:
            sse = 0.0
            for region in range(3):
                for risk in range(2):
                    sse += np.sum(np.square(run_data[:, 1+6*region+3*risk:2+6*region+3*risk] -
                                            self.data['dpc_inf'][region, risk, :, :]))
            fitter.assessment['sse'] = sse
        else:
            sse = 0.0
            for region in range(3):
                for risk in range(2):
                    sse += np.sum(np.square(run_data[:, 1+6*region+3*risk:2+6*region+3*risk] -
                                            self.data['controlled_dpc_inf'][region, risk, :, :]))
            fitter.assessment['sse'] = sse
            fitter.assessment['controlled_sse'] = sse

        # Calculate AIC
        if control is None:
            aic = 2*12
            aic -= 2*self.linear_fits['likelihood']._calc_likelihood(
                params=fitter.data, events=self.data['assessment_events'])
            fitter.assessment['aic'] = aic

        else:
            aic = 2*12
            aic -= 2*self.linear_fits['likelihood']._calc_likelihood(
                params=fitter.data, events=self.data['controlled_assessment_events'])
            fitter.assessment['controlled_aic'] = aic

    def _likelihood_map(self, fitter, save_folder=None):
        """Generate heat map of likelihood surface."""

        parameters = {"sigma"+str(i)+str(j): ("sigma", (i, j)) for i in range(3) for j in range(3)}
        parameters["rho01"] = ("rho", (0, 1))
        parameters["rho10"] = ("rho", (1, 0))
        parameters["rho11"] = ("rho", (1, 1))

        if save_folder is None:
            pdf_name = fitter.name + "_LikelihoodSurface.pdf"
        else:
            pdf_name = os.path.join(save_folder, fitter.name + "_LikelihoodSurface.pdf")

        with PdfPages(pdf_name) as pdf:
            parameter_set = itertools.combinations(parameters.keys(), 2)

            for figure_num, (param1, param2) in enumerate(parameter_set):
                if figure_num % 9 == 0:
                    fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27))
                    axes = axes.flatten()

                test_params = {
                    'sigma': copy.deepcopy(fitter.data['sigma']),
                    'rho': copy.deepcopy(fitter.data['rho'])
                }

                p1_name, p1_idx = parameters[param1]
                p2_name, p2_idx = parameters[param2]

                x_vals = np.linspace(0.1*test_params[p1_name][p1_idx],
                                     10*test_params[p1_name][p1_idx], 21)
                y_vals = np.linspace(0.1*test_params[p2_name][p2_idx],
                                     10*test_params[p2_name][p2_idx], 21)

                xx, yy = np.meshgrid(x_vals, y_vals)
                zz = np.zeros_like(xx)

                for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
                    test_params[p1_name][p1_idx] = xx[i, j]
                    test_params[p2_name][p2_idx] = yy[i, j]
                    zz[i, j] = self.linear_fits['likelihood']._calc_likelihood(params=test_params)

                im = axes[figure_num % 9].pcolormesh(xx, yy, zz)
                divider = make_axes_locatable(axes[figure_num % 9])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, ax=axes[figure_num % 9], cax=cax)

                axes[figure_num % 9].plot(fitter.data[p1_name][p1_idx],
                                          fitter.data[p2_name][p2_idx], "rx", ms=3)

                axes[figure_num % 9].set_xlabel(param1)
                axes[figure_num % 9].set_ylabel(param2)

                if figure_num % 9 == 8:
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

class SpaceFitterLikelihood:
    """Class for fitting space structured OCT model to simulations using maximum likelihood."""

    def __init__(self, parent_fitter):
        self.parent_fitter = parent_fitter

        self.name = "Linear_Likelihood"
        self.data = {}
        self.assessment = {}

    def __repr__(self):
        repr_str = " ".join(self.name.split("_")) + "Fit\n"
        if self.data:
            repr_str += "Sigma: {0}\n".format(self.data['sigma'])
            repr_str += "Rho: {0}\n".format(self.data['rho'])

        if self.assessment:
            for key, val in sorted(self.assessment.items()):
                repr_str += "{0}: {1}\n".format(key, str(val))

        repr_str += "\n" + "-"*20 + "\n\n"

        return repr_str

    def fit(self, bounds=None, start=None):
        """Fit incidence function parameters."""

        nparams = 8

        start_vals = np.zeros(nparams)
        bound_vals = np.zeros((nparams, 2))

        start_vals[0:5] = start['sigma'].flatten()[[0, 3, 4, 7, 8]]
        bound_vals[0:5, :] = bounds['sigma'].reshape(9, 2)[[0, 3, 4, 7, 8]]

        start_vals[5:] = [start['rho'][0, 1], start['rho'][1, 0], start['rho'][1, 1]]
        bound_vals[5:, :] = bounds['rho'].reshape(4, 2)[[1, 2, 3]]

        param_dict_tmp = {
            'sigma': np.zeros((3, 3)),
            'rho': np.zeros((2, 2))
        }
        param_dict_tmp['rho'][0, 0] = 1
        param_dict_tmp['sigma'][0, 2] = 0
        param_dict_tmp['sigma'][2, 0] = 0
        param_dict_tmp['sigma'][0, 1] = 0
        param_dict_tmp['sigma'][1, 0] = 0

        start_transformed = fitting.logit_transform(start_vals, bound_vals)

        def neg_loglik(params, bounds, param_dict):
            """Calculate neg log likelihood from transformed optimisation parameters."""
            # First reverse logit transform parameters
            _params_rev_trans = fitting.reverse_logit_transform(params, bounds)
            param_dict['sigma'][[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]] = _params_rev_trans[0:5]
            param_dict['rho'][[0, 1, 1], [1, 0, 1]] = _params_rev_trans[5:]

            val = self._calc_likelihood(param_dict)
            return np.nan_to_num(-val)

        param_fit_transformed = minimize(
            neg_loglik, start_transformed, method="L-BFGS-B", options={'ftol': 1e-12},
            args=(bound_vals, param_dict_tmp))

        param_fit = fitting.reverse_logit_transform(param_fit_transformed.x, bound_vals)

        if 'sigma' not in self.data:
            self.data['sigma'] = np.zeros((3, 3))
        if 'rho' not in self.data:
            self.data['rho'] = np.ones((2, 2))

        self.data['sigma'][[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]] = param_fit[0:5]
        self.data['rho'][[0, 1, 1], [1, 0, 1]] = param_fit[5:]

    def predict_rates(self, states, params=None):
        """Calculate predicted infection rates.

        Columns in returned array give rates 1HIGH, 1LOW, 2HIGH, 2LOW, 3HIGH, 3LOW.
        """

        masked = np.ma.is_masked(states)
        states = np.clip(states, 0, 2e10).reshape((-1, 12))

        if params is None:
            params = self.data

        rates = np.ma.zeros((len(states), 6))
        sigma = params['sigma']
        rho = params['rho']
        for region1, risk1, region2, risk2 in itertools.product(range(3), range(2), repeat=2):
            rates[:, 2*region1+risk1] += (
                sigma[region1, region2] * rho[risk1, risk2] * states[:, 4*region1+2*risk1] *
                states[:, 1+4*region2+2*risk2])

        rates.set_fill_value(0)
        if not masked:
            rates = np.ma.filled(rates)

        return rates

    def _calc_likelihood(self, params=None, events=None):
        """Calculate log likelihood value for given parameters and list of risks and regions."""

        if "input_events" not in self.parent_fitter.parent_fitter.data:
            raise ValueError("Run data intialisation first!")

        if events is None:
            events = self.parent_fitter.parent_fitter.data['input_events']

        log_lik = 0

        for run in events:
            rates = self.predict_rates(run[:, 0:12], params)

            log_lik -= np.sum(rates * run[:, 12:13])
            idcs = np.where(run[:, 13] > 0)[0]
            log_lik += np.sum(np.log(rates[idcs,
                                           np.array(2*run[idcs, 15]+run[idcs, 14], dtype=int)]))

        return log_lik

class SpaceFitterSSE:
    """Class for fitting space structured OCT model to sims by minimising sum squared errors."""

    def __init__(self, parent_fitter):
        self.parent_fitter = parent_fitter

        self.name = "Linear_Likelihood"
        self.data = {}
        self.assessment = {}

    def __repr__(self):
        repr_str = " ".join(self.name.split("_")) + "Fit\n"
        if self.data:
            repr_str += "Sigma: {0}\n".format(self.data['sigma'])
            repr_str += "Rho: {0}\n".format(self.data['rho'])

        if self.assessment:
            for key, val in sorted(self.assessment.items()):
                repr_str += "{0}: {1}\n".format(key, str(val))

        repr_str += "\n" + "-"*20 + "\n\n"

        return repr_str

    def sse(self, params, bounds, model):
        """Calculate sum squared errors from transformed optimisation parameters."""
        # First reverse logit transform parameters
        _params_rev_trans = fitting.reverse_logit_transform(params, bounds)
        self.data['sigma'][[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]] = _params_rev_trans[0:5]
        self.data['rho'][[0, 1, 1], [1, 0, 1]] = _params_rev_trans[5:]

        # Run ODE model with parameters
        ode_run = model.run_policy(space_model.no_control_policy)

        dpc_data = self.parent_fitter.parent_fitter.data['dpc_inf']
        dpc_times = self.parent_fitter.parent_fitter.data['dpc_times']

        # Calculate SSE
        sse = 0
        for i in range(dpc_data.shape[3]):
            for region in range(3):
                sse += np.sum(np.square(
                    dpc_data[region, 0, :, i] -
                    np.array([ode_run.state(t)[6*region+1] for t in dpc_times])))
                sse += np.sum(np.square(
                    dpc_data[region, 1, :, i] -
                    np.array([ode_run.state(t)[6*region+4] for t in dpc_times])))
        return sse

    def fit(self, bounds=None, start=None):
        """Fit incidence function parameters."""

        nparams = 8

        start_vals = np.zeros(nparams)
        bound_vals = np.zeros((nparams, 2))

        start_vals[0:5] = start['sigma'].flatten()[[0, 3, 4, 7, 8]]
        bound_vals[0:5, :] = bounds['sigma'].reshape(9, 2)[[0, 3, 4, 7, 8]]

        start_vals[5:] = [start['rho'][0, 1], start['rho'][1, 0], start['rho'][1, 1]]
        bound_vals[5:, :] = bounds['rho'].reshape(4, 2)[[1, 2, 3]]

        self.data['sigma'] = np.zeros((3, 3), dtype=float)
        self.data['rho'] = np.ones((2, 2), dtype=float)

        start_transformed = fitting.logit_transform(start_vals, bound_vals)

        init_state = self.parent_fitter.parent_fitter.data['init_state'][0]

        model_params = {
            'birth_rate': self.parent_fitter.parent_fitter.sim_params['birth_rate'],
            'death_rate': self.parent_fitter.parent_fitter.sim_params['death_rate'],
            'removal_rate': self.parent_fitter.parent_fitter.sim_params['removal_rate'],
            'recov_rate': self.parent_fitter.parent_fitter.sim_params['recov_rate'],
            'state_init': init_state,
            'times': self.parent_fitter.parent_fitter.data['dpc_times'],
            'max_control_rate': 0,
            'high_alloc_cost': 0,
            'low_alloc_cost': 0
        }
        model = space_model.SpaceModel(model_params, self)

        param_fit_transformed = minimize(
            self.sse, start_transformed, method="L-BFGS-B", options={'ftol': 1e-12},
            args=(bound_vals, model))

        param_fit = fitting.reverse_logit_transform(param_fit_transformed.x, bound_vals)

        if 'sigma' not in self.data:
            self.data['sigma'] = np.zeros((3, 3), dtype=float)
        if 'rho' not in self.data:
            self.data['rho'] = np.ones((2, 2), dtype=float)

        self.data['sigma'][[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]] = param_fit[0:5]
        self.data['rho'][[0, 1, 1], [1, 0, 1]] = param_fit[5:]

    def predict_rates(self, states, params=None):
        """Calculate predicted infection rates.

        Columns in returned array give rates 1HIGH, 1LOW, 2HIGH, 2LOW, 3HIGH, 3LOW.
        """

        masked = np.ma.is_masked(states)
        states = np.clip(states, 0, 2e10).reshape((-1, 12))

        if params is None:
            params = self.data

        rates = np.ma.zeros((len(states), 6))
        sigma = params['sigma']
        rho = params['rho']
        for region1, risk1, region2, risk2 in itertools.product(range(3), range(2), repeat=2):
            rates[:, 2*region1+risk1] += (
                sigma[region1, region2] * rho[risk1, risk2] * states[:, 4*region1+2*risk1] *
                states[:, 1+4*region2+2*risk2])

        rates.set_fill_value(0)
        if not masked:
            rates = np.ma.filled(rates)

        return rates
