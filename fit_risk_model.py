"""Code for parameter and function fitting of risk based approximate model to simulation data."""

import pdb
import copy
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import simulation
from simulation import Risk
import risk_model
import control_tools
import fitting
import visualisation

class RiskFitter:
    """Class for fitting risk based approximate model, handling likelihood and SSE fits."""

    def __init__(self, parent):
        self.data = {}
        self.linear_fits = {}
        self.parent_fitter = parent

    def __repr__(self):
        repr_str = "\nRisk Model Fitter\n" + "-"*20 + "\n\n"

        if self.linear_fits:
            for _, fit in self.linear_fits.items():
                repr_str += repr(fit)
        else:
            repr_str += "No linear fits to show\n" + "-"*20 + "\n\n"

        return repr_str

    def fit(self, bounds, start):
        """Fit risk model to generated data."""

        input_events = self.parent_fitter.data['input_events']
        risk_input_events = []

        # Convert form of input_events to be risk based only (no spatial info)
        for run in input_events:
            risk_run = np.zeros((len(run), 7))
            for i in range(4):
                risk_run[:, i] = np.sum(run[:, i:12:4], axis=1)
            risk_run[:, 4:] = run[:, 12:15]
            risk_input_events.append(risk_run)

        self.data['input_events'] = risk_input_events

        self.linear_fits['likelihood'] = RiskFitterLikelihood(self)
        self.linear_fits['likelihood'].fit(bounds, start)

        print("{0} fit complete.".format(self.linear_fits['likelihood'].name))

    def assess(self, fitter, save_folder=None, control=None, likelihood_map=False,
               initial_nodes=None, max_control_rate=100):
        """Assess fit quality."""

        if "init_conds" not in self.data:
            self.optimise(fitter, initial_nodes)

        print("Assessing {0} risk model...".format(fitter.name))

        # Generate model run data
        if control is None:
            controller = risk_model.no_control_policy
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
        model = risk_model.RiskModel(model_params, fitter)
        model_run = model.run_policy(controller)
        run_data = np.array([model_run.state(t) for t in model_params['times']])

        self._assess_dpc(run_data, control=control, save_folder=save_folder)

        self._calc_metrics(fitter, run_data, control=control)

        if likelihood_map:
            self._likelihood_map(fitter, save_folder=save_folder)

    def optimise(self, fitter, initial_nodes=None, max_control_rate=100, run_sims=True):
        """Optimise control on given fit and run associated simulations."""

        print("Optimising risk based control...")

        if initial_nodes is None:
            nodes_init = fitting.randomise_infection(self.parent_fitter.nodes, nrand=5)
        else:
            nodes_init = copy.deepcopy(initial_nodes)

        state_init = np.zeros(6)

        for node in nodes_init:
            states = sorted(simulation.HIGH_LIVE_STATES | simulation.LOW_LIVE_STATES)
            for i, state_val in enumerate(states):
                state_init[i] += node.state[state_val]

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
        model = risk_model.RiskModel(model_params, fitter)
        bocop_run = model.run_bocop(verbose=False, init_policy=risk_model.even_control_policy)
        opt_control = bocop_run.control
        self.data["opt_control"] = opt_control
        self.data["init_conds"] = state_init
        self.data["dpc_times"] = model_params['times']

        if run_sims:
            self._get_sim_dpcs(model, opt_control, initial_nodes)

    def _get_sim_dpcs(self, model, opt_control, initial_nodes=None):
        """Run simulations with and without control for assessment DPCs."""

        sim_params = copy.deepcopy(self.parent_fitter.sim_params)

        # Run no control sims
        print("Generating risk testing set...")
        stored_data = {}
        input_events = []
        for _ in range(10):
            if initial_nodes is None:
                nodes_init = fitting.randomise_infection(
                    self.parent_fitter.nodes, nhigh=self.data['init_conds'][1],
                    nlow=self.data['init_conds'][4])
            else:
                nodes_init = copy.deepcopy(initial_nodes)

            simulator = simulation.Simulator(nodes_init, self.parent_fitter.risk_coupling,
                                             self.parent_fitter.dist_coupling, sim_params,
                                             controller=None)
            all_runs = simulator.run_simulation(nruns=10, verbose=False)
            self.parent_fitter._store_dpc_data(all_runs, data_store=stored_data, include_vac=False)
            input_events += self.parent_fitter._extract_event_data(all_runs)

        # Convert to risk based format
        risk_events = []
        for run in input_events:
            risk_run = np.zeros((len(run), 7))
            for i in range(4):
                risk_run[:, i] = np.sum(run[:, i:12:4], axis=1)
            risk_run[:, 4:] = run[:, 12:15]
            risk_events.append(risk_run)

        for key, val in stored_data.items():
            self.data[key] = val
        self.data["assessment_events"] = risk_events

        sim_params['update_control_on_all_events'] = True
        sim_params['vac_rate'] = 1.0
        sim_params["controller_args"] = {
            "oc_model": model,
            "policy": opt_control
        }

        # Run controlled simulations
        print("Generating controlled risk testing set...")
        stored_data = {}
        input_events = []
        for i in range(10):
            if initial_nodes is None:
                nodes_init = fitting.randomise_infection(
                    self.parent_fitter.nodes, nhigh=self.data['init_conds'][1],
                    nlow=self.data['init_conds'][4])
            else:
                nodes_init = copy.deepcopy(initial_nodes)

            simulator = simulation.Simulator(nodes_init, self.parent_fitter.risk_coupling,
                                             self.parent_fitter.dist_coupling, sim_params,
                                             controller=control_tools.RiskPolicyController)
            all_runs = simulator.run_simulation(nruns=10, verbose=False)
            self.parent_fitter._store_dpc_data(all_runs, data_store=stored_data, include_vac=True)
            input_events += self.parent_fitter._extract_event_data(all_runs)

        # Convert to risk based format
        risk_events = []
        for run in input_events:
            risk_run = np.zeros((len(run), 7))
            for i in range(4):
                risk_run[:, i] = np.sum(run[:, i:12:4], axis=1)
            risk_run[:, 4:] = run[:, 12:15]
            risk_events.append(risk_run)

        for key, val in stored_data.items():
            self.data["controlled_" + key] = val
        self.data["controlled_assessment_events"] = risk_events

    def _assess_dpc(self, run_data, control=None, save_folder=None, save_name="Linear_Likelihood"):
        """Plot DPC data fit against simulation data."""

        data = self.data

        fig, axes = plt.subplots(1, 2)
        for risk in range(2):
            if control is None:
                axes[risk].plot(data['dpc_times'], np.sum(data['dpc_sus'][:, risk, :, :], axis=0),
                                'g--', alpha=0.1)
                axes[risk].plot(data['dpc_times'], np.sum(data['dpc_inf'][:, risk, :, :], axis=0),
                                'r--', alpha=0.1)

            else:
                axes[risk].plot(data['dpc_times'],
                                np.sum(data['controlled_dpc_sus'][:, risk, :, :], axis=0),
                                'g--', alpha=0.1)
                axes[risk].plot(data['dpc_times'],
                                np.sum(data['controlled_dpc_inf'][:, risk, :, :], axis=0),
                                'r--', alpha=0.1)
                axes[risk].plot(data['dpc_times'],
                                np.sum(data['controlled_dpc_vac'][:, risk, :, :], axis=0),
                                '--', color="purple", alpha=0.1)

            axes[risk].plot(data['dpc_times'], run_data[:, 3*risk], 'g-', lw=2)
            axes[risk].plot(data['dpc_times'], run_data[:, 1+3*risk], 'r-', lw=2)

            if control is not None:
                axes[risk].plot(data['dpc_times'], run_data[:, 2+3*risk], '-', color="purple", lw=2)

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

        fig, axes = plt.subplots(1, 2)
        for risk in range(2):
            if control is None:
                visualisation.err_bands(np.sum(data['dpc_inf'][:, risk, :, :], axis=0),
                                        axes[risk], data['dpc_times'], col="red",
                                        alpha_range=[0.05, 0.35],
                                        lower_percentiles=np.linspace(0, 50, 21, endpoint=False))

            else:
                visualisation.err_bands(
                    np.sum(data['controlled_dpc_inf'][:, risk, :, :], axis=0), axes[risk],
                    data['dpc_times'], col="red", alpha_range=[0.05, 0.35],
                    lower_percentiles=np.linspace(0, 50, 21, endpoint=False))

            axes[risk].plot(data['dpc_times'], run_data[:, 1+3*risk], 'r-', lw=2)

        if control is None:
            fig_name = save_name + "_median.pdf"
        else:
            fig_name = save_name + "_Controlled_median.pdf"


        fig.tight_layout()
        if save_folder is None:
            fig.savefig(fig_name)
        else:
            fig.savefig(os.path.join(save_folder, fig_name))
        plt.close(fig)

    def _calc_metrics(self, fitter, run_data, control=None):
        """Calulate model selection metrics."""

        # Calculate SSE
        sse = 0.0
        if control is None:
            for risk in range(2):
                sse += np.sum(np.square(run_data[:, 1+3*risk:2+3*risk] -
                                        np.sum(self.data['dpc_inf'][:, risk, :, :], axis=0)))
            fitter.assessment['sse'] = sse
        else:
            for risk in range(2):
                sse += np.sum(np.square(run_data[:, 1+3*risk:2+3*risk] - np.sum(
                    self.data['controlled_dpc_inf'][:, risk, :, :], axis=0)))
            fitter.assessment['controlled_sse'] = sse

        # Calculate AIC
        if control is None:
            aic = 2*4
            aic -= 2*self.linear_fits['likelihood']._calc_likelihood(
                params=fitter.data, events=self.data['assessment_events'])
            fitter.assessment['aic'] = aic

        else:
            aic = 2*4
            aic -= 2*self.linear_fits['likelihood']._calc_likelihood(
                params=fitter.data, events=self.data['controlled_assessment_events'])
            fitter.assessment['controlled_aic'] = aic

    def _likelihood_map(self, fitter, save_folder=None):
        """Generate heat map of likelihood surface."""

        fig, axes = plt.subplots(1, 2)

        for risk_1 in Risk:
            risk_2 = int(not bool(risk_1))
            beta = fitter.data['beta']

            x_vals = np.linspace(0.1*beta[risk_1, risk_1], 10*beta[risk_1, risk_1], 51)
            y_vals = np.linspace(0.1*beta[risk_1, risk_2], 10*beta[risk_1, risk_2], 51)

            xx, yy = np.meshgrid(x_vals, y_vals)
            zz = np.zeros_like(xx)

            test_beta = np.zeros((2, 2))
            test_beta[risk_2] = beta[risk_2]
            test_params = {'beta': test_beta}

            for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
                test_beta[risk_1, risk_1] = xx[i, j]
                test_beta[risk_1, risk_2] = yy[i, j]
                zz[i, j] = self.linear_fits['likelihood']._calc_likelihood(
                    params=test_params, events=self.data['assessment_events'])

            cmap = plt.get_cmap("inferno")
            vmax = np.max(zz)
            im = axes[risk_1].pcolormesh(xx, yy, zz, cmap=cmap, vmin=0.98*vmax, vmax=vmax)

            divider = make_axes_locatable(axes[risk_1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=axes[risk_1], cax=cax)

            axes[risk_1].plot(beta[risk_1, risk_1], beta[risk_1, risk_2], "rx", ms=3)

            axes[risk_1].set_xlabel(r"$\beta_{{{0}}}$".format(str(risk_1)[5] + str(risk_1)[5]))
            axes[risk_1].set_ylabel(r"$\beta_{{{0}}}$".format(str(risk_1)[5] +
                                                              str(simulation.Risk(risk_2))[5]))

        fig.tight_layout()
        fig_name = fitter.name + "_LikelihoodSurface.pdf"

        if save_folder is None:
            fig.savefig(fig_name)
        else:
            fig.savefig(os.path.join(save_folder, fig_name))
        plt.close(fig)

class RiskFitterLikelihood:
    """Class for fitting risk structured OCT model to simulations using maximum likelihood."""

    def __init__(self, parent_fitter):
        self.parent_fitter = parent_fitter

        self.name = "Linear_Likelihood"
        self.data = {}
        self.assessment = {}

    def __repr__(self):
        repr_str = " ".join(self.name.split("_")) + "Fit\n"
        if self.data:
            repr_str += "Beta: {0}\n".format(self.data['beta'])

        if self.assessment:
            for key, val in sorted(self.assessment.items()):
                repr_str += "{0}: {1}\n".format(key, str(val))

        repr_str += "\n" + "-"*20 + "\n\n"

        return repr_str

    def fit(self, bounds=None, start=None):
        """Fit incidence function parameters."""

        # Fit parameters associated with each risk group separately since log likelihood separates
        for risk in Risk:
            start_vals = start['beta'][risk]
            bound_vals = bounds['beta'][risk]

            param_dict_tmp = {
                'beta': np.zeros((2, 2))
            }

            start_transformed = fitting.logit_transform(start_vals, bound_vals)

            def neg_loglik(params, risk, bounds, param_dict):
                """Calculate negative log likelihood from transformed optimisation parameters."""
                # First reverse logit transform parameters
                _params_rev_trans = fitting.reverse_logit_transform(params, bounds)
                param_dict['beta'][risk] = _params_rev_trans
                val = self._calc_likelihood(param_dict, [risk])

                return np.nan_to_num(-val)

            # Minimise negative log likelihood
            param_fit_transformed = minimize(
                neg_loglik, start_transformed, method="L-BFGS-B", options={'ftol': 1e-12},
                args=(risk, bound_vals, param_dict_tmp))

            param_fit = fitting.reverse_logit_transform(param_fit_transformed.x, bound_vals)

            if 'beta' not in self.data:
                self.data['beta'] = np.zeros((2, 2))

            self.data['beta'][risk] = param_fit

    def predict_rates(self, states, params=None):
        """Calculate predicted infection rates.
        
        Arguments:
            states:     Array of SH, IH, SL, IL (with multiple rows if calculating multiple rates
            params:     Dictionary to find beta values. If None uses self.data
        """

        masked = np.ma.is_masked(states)
        states = np.clip(states, 0, 2e10).reshape((-1, 4))

        if params is None:
            params = self.data

        beta = params['beta']
        rates = np.ma.array([
            beta[i//2, i%2] *
            np.prod(states[:, [2*(i//2), 2*(i%2)+1]], axis=1) for i in range(4)]).T

        rates.set_fill_value(0)
        if not masked:
            rates = np.ma.filled(rates)

        return rates

    def _calc_likelihood(self, params=None, risks=None, events=None):
        """Calculate log likelihood value for given parameters and given list of risks."""

        if "input_events" not in self.parent_fitter.parent_fitter.data:
            raise ValueError("Run data intialisation first!")

        if risks is None:
            risks = [Risk.HIGH, Risk.LOW]

        if events is None:
            events = self.parent_fitter.data['input_events']
        log_lik = 0

        for run in events:
            states = run[:, 0:4]
            delta_t = run[:, 4]
            inf_multi = run[:, 5]
            inf_risk = run[:, 6]

            all_rates = self.predict_rates(states, params)
            # log_lik = lambda_k * exp(sum(lambda_i*deltat)) for each event
            for risk in risks:
                rates = np.sum(all_rates[:, 2*risk:(2*risk+2)], axis=1)
                log_lik -= np.sum(rates * delta_t)
                idcs = np.where((inf_multi > 0) & (inf_risk == risk))[0]
                log_lik += np.sum(np.log(rates[idcs]))

        return log_lik

class RiskFitterSSE:
    """Class for fitting risk structured OCT model to sims by minimising sum squared errors."""

    def __init__(self, parent_fitter):
        self.parent_fitter = parent_fitter

        self.name = "Linear_SSE"
        self.data = {}
        self.assessment = {}

    def __repr__(self):
        repr_str = " ".join(self.name.split("_")) + "Fit\n"
        if self.data:
            repr_str += "Beta: {0}\n".format(self.data['beta'])

        if self.assessment:
            for key, val in sorted(self.assessment.items()):
                repr_str += "{0}: {1}\n".format(key, str(val))

        repr_str += "\n" + "-"*20 + "\n\n"

        return repr_str

    def sse(self, params, bounds, model):
        """Calculate sum squared errors from transformed optimisation parameters."""
        # First reverse logit transform parameters
        _params_rev_trans = fitting.reverse_logit_transform(params, bounds)
        self.data['beta'] = np.array(_params_rev_trans).reshape((2, 2))

        # Run ODE model with parameters
        ode_run = model.run_policy(risk_model.no_control_policy)

        dpc_data = self.parent_fitter.parent_fitter.data['dpc_inf']
        dpc_times = self.parent_fitter.parent_fitter.data['dpc_times']

        # Calculate SSE
        sse = 0
        for i in range(dpc_data.shape[3]):
            sse += np.sum(np.square(np.sum(
                dpc_data[:, 0, :, i], axis=0) - np.array([ode_run.state(t)[1] for t in dpc_times])))
            sse += np.sum(np.square(np.sum(
                dpc_data[:, 1, :, i], axis=0) - np.array([ode_run.state(t)[4] for t in dpc_times])))
        return sse

    def fit(self, bounds=None, start=None):
        """Fit incidence function parameters."""

        start_vals = start['beta'].flatten()
        bound_vals = bounds['beta'].flatten().reshape((4, 2))

        start_transformed = fitting.logit_transform(start_vals, bound_vals)

        init_state = np.sum(self.parent_fitter.parent_fitter.data['init_state'][0].reshape((3, 6)),
                            axis=0)

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
        model = risk_model.RiskModel(model_params, self)

        # Minimise SSE
        param_fit_transformed = minimize(
            self.sse, start_transformed, method="L-BFGS-B", options={'ftol': 1e-12},
            args=(bound_vals, model))

        param_fit = fitting.reverse_logit_transform(param_fit_transformed.x, bound_vals)

        self.data['beta'] = np.array(param_fit).reshape((2, 2))

    def predict_rates(self, states, params=None):
        """Calculate predicted infection rates."""

        masked = np.ma.is_masked(states)
        states = np.clip(states, 0, 2e10).reshape((-1, 4))

        if params is None:
            params = self.data

        beta = params['beta']
        rates = np.ma.array([
            beta[i//2, i%2] *
            np.prod(states[:, [2*(i//2), 2*(i%2)+1]], axis=1) for i in range(4)]).T

        rates.set_fill_value(0)
        if not masked:
            rates = np.ma.filled(rates)

        return rates
