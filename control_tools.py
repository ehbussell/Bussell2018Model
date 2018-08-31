"""Controller objects for vaccination control in simulations."""

import warnings
import time as time_mod
import numpy as np
from simulation import Event, Risk
import risk_model
import space_model

class MPCRiskController:
    """Risk structured controller for simulation using MPC algorithm to allocate resources.

    Initialisation requires:
        oc_model:       Optimal control model (risk based) to use for optimisation.
        mpc_params:     Dictionary of parameters controlling MPC setup. Contains:
                            update_freq:            Time between re-optimisation of oc_model
                            model_horizon:          Time horizon for optimisation
                            rolling_horizon:        Boolean whether horizon should move forward as
                                                    simulation progresses. Default: True
                            init_policy:            Optional policy to use to initialise
                                                    optimisation. Default: None
                            initial_control_func:   Optional control function to use initially. Can
                                                    be used avoid repeated identical optimisation.
                                                    Function must return (HIGH, LOW) tuple.
                                                    Default: None
    """

    def __init__(self, oc_model, mpc_params, verbose=False):

        self.model = oc_model
        self.update_freq = mpc_params['update_freq']
        self.model_horizon = mpc_params['model_horizon']
        self.rolling = mpc_params.get('rolling_horizon', True)
        self.init_policy = mpc_params.get('init_policy', None)

        self.current_control = mpc_params.get('initial_control_func', None)
        self.next_renew_time = 0

        if self.model_horizon is not None:
            self.delta_t = self.model_horizon / 100
        else:
            self.delta_t = np.nan

        self.verbose = verbose

        if self.current_control is not None:
            self.log = [(0.0, self.model.run_policy(mpc_params['initial_control_func']))]
        else:
            self.log = []

    def __call__(self, time, nodes, run_data, new_events, vac_rates, scheduled=False):

        if self.current_control is None:
            self.renew_controller(time, run_data)

        if scheduled and np.isclose(time, self.next_renew_time):
            self.renew_controller(time, run_data)

        return self.reallocate(time, run_data)

    def reallocate(self, time, run_data):
        """Reallocate resources as given by oc_model."""

        new_rates_ret = []
        new_events_ret = []
        new_reg_vac_factors = []

        state = np.array(run_data["Global"][-1])[[1, 2, 3, 5, 6, 7]]

        control = self.current_control(time)

        if np.sum(control) > 1.001:
            warnings.warn("Budget exceeded by {0}%!".format(100*(np.sum(control)-1.0)))
            control = control / np.sum(control)

        nhigh = np.sum(state[0:3])
        nlow = np.sum(state[3:6])

        new_rate_factors = {
            Event.VAC_H: (control[0] * self.model.params['max_control_rate'] /
                          nhigh if nhigh > 0 else 0),
            Event.VAC_L: (control[1] * self.model.params['max_control_rate'] /
                          nlow if nlow > 0 else 0),
        }

        return new_rates_ret, new_events_ret, new_rate_factors, new_reg_vac_factors

    def renew_controller(self, time, run_data):
        """Re-run oc_model to update resource allocation."""

        if self.verbose:
            print("Renewing controller at time {0}".format(time))

        state = np.array(run_data["Global"][-1])[[1, 2, 3, 5, 6, 7]]

        if self.rolling:
            times = np.linspace(time, time + self.model_horizon, 101)
        else:
            # times = np.linspace(time, self.model_horizon, 101)
            times = np.arange(time, self.model_horizon, self.delta_t)
        self.model.params['state_init'] = state
        self.model.params['times'] = times

        if state[1] == 0 and state[4] == 0:
            self.current_control = risk_model.no_control_policy
            self.next_renew_time += self.update_freq
            self.log.append((time, self.model.run_policy(risk_model.no_control_policy)))
            return

        bocop_run = self.model.run_bocop(verbose=False, init_policy=self.init_policy)

        if self.verbose:
            print(bocop_run.exit_code)
        if bocop_run.exit_code != "Optimal Solution Found.":
            warnings.warn("Convergence Failure!")

        self.current_control = bocop_run.control
        self.next_renew_time += self.update_freq

        self.log.append((time, bocop_run))

    def get_log(self):
        """Return log of optimiser outputs at update times."""

        return self.log

class RiskPolicyController(MPCRiskController):
    """Risk structured open loop controller using given policy to allocate resources."""

    def __init__(self, oc_model, policy):
        mpc_params = {
            'update_freq': np.inf,
            'model_horizon': None,
            'rolling_horizon': False,
            'init_policy': None,
            'initial_control_func': policy
        }

        super().__init__(oc_model, mpc_params)

class MPCSpaceController:
    """Space structured controller for simulation using MPC algorithm to allocate resources.

    Initialisation requires:
        oc_model:       Optimal control model (space based) to use for optimisation.
        mpc_params:     Dictionary of parameters controlling MPC setup. Contains:
                            update_freq:            Time between re-optimisation of oc_model
                            model_horizon:          Time horizon for optimisation
                            rolling_horizon:        Boolean whether horizon should move forward as
                                                    simulation progresses. Default: True
                            cold_start_files:       List of .sol files to use for cold start of
                                                    optimisation. Will attempt to use one per
                                                    optimisation, with fall back to init_policy.
                                                    Default: None
                            init_policy:            Optional policy to use to initialise
                                                    optimisation. Default: None
                            initial_control_func:   Optional control function to use initially. Can
                                                    be used to avoid repeated identical
                                                    optimisation. Function must return
                                                    (HA, LA, HB, LB, HC, LC) or (A, B, C) tuple.
                                                    Default: None
    """

    def __init__(self, oc_model, mpc_params, verbose=False):

        self.model = oc_model
        self.update_freq = mpc_params['update_freq']
        self.model_horizon = mpc_params['model_horizon']
        self.rolling = mpc_params.get('rolling_horizon', True)
        self.cold_start_files = iter(mpc_params.get('cold_start_files', []))
        self.init_policy = mpc_params.get('init_policy', None)

        self.current_control = mpc_params.get('initial_control_func', None)
        self.next_renew_time = 0

        if self.model_horizon is not None:
            self.delta_t = self.model_horizon / 100
        else:
            self.delta_t = np.nan
        self.verbose = verbose

        if self.current_control is not None:
            self.log = [(0.0, self.model.run_policy(mpc_params['initial_control_func']))]
        else:
            self.log = []

    def __call__(self, time, nodes, run_data, new_events, vac_rates, scheduled=False):

        if self.current_control is None:
            self.renew_controller(time, run_data)

        if scheduled and np.isclose(time, self.next_renew_time):
            self.renew_controller(time, run_data)

        return self.reallocate(time, run_data)

    def reallocate(self, time, run_data):
        """Reallocate resources as given by oc_model."""

        new_rates_ret = []
        new_events_ret = []
        new_rate_factors = {}
        new_reg_vac_factors = []

        state = np.zeros(18)
        state[0:6] = np.array(run_data["RegionA"][-1])[[1, 2, 3, 5, 6, 7]]
        state[6:12] = np.array(run_data["RegionB"][-1])[[1, 2, 3, 5, 6, 7]]
        state[12:18] = np.array(run_data["RegionC"][-1])[[1, 2, 3, 5, 6, 7]]

        control_treat = self.current_control(time)
        if np.sum(control_treat) - 1.0 > 1e-5:
            warnings.warn("Budget exceeded by {0}%!".format(100*(np.sum(control_treat)-1.0)))
            control_treat = control_treat / np.sum(control_treat)

        if len(control_treat) == 3:
            for i, region in enumerate(["A", "B", "C"]):
                nhigh = np.sum(state[6*i:(6*i+3)])
                nlow = np.sum(state[(6*i+3):(6*i+6)])
                treat = [control_treat[i] * nhigh / (nhigh + nlow),
                         control_treat[i] * nlow / (nhigh + nlow)]
                new_reg_vac_factors.append((
                    region, Risk.HIGH, treat[0] * self.model.params['max_control_rate'] /
                    nhigh if nhigh > 0 else 0
                ))
                new_reg_vac_factors.append((
                    region, Risk.LOW, treat[1] * self.model.params['max_control_rate'] /
                    nlow if nlow > 0 else 0
                ))

        else:
            for i, region in enumerate(["A", "B", "C"]):
                nhigh = np.sum(state[6*i:(6*i+3)])
                nlow = np.sum(state[(6*i+3):(6*i+6)])
                new_reg_vac_factors.append((
                    region, Risk.HIGH, control_treat[2*i] * self.model.params['max_control_rate'] /
                    nhigh if nhigh > 0 else 0
                ))
                new_reg_vac_factors.append((
                    region, Risk.LOW, control_treat[1+2*i] * self.model.params['max_control_rate'] /
                    nlow if nlow > 0 else 0
                ))

        return new_rates_ret, new_events_ret, new_rate_factors, new_reg_vac_factors

    def renew_controller(self, time, run_data):
        """Re-run oc_model to update resource allocation."""

        if self.verbose:
            print("Renewing controller at time {0}".format(time))

        state = np.zeros(18)
        state[0:6] = np.array(run_data["RegionA"][-1])[[1, 2, 3, 5, 6, 7]]
        state[6:12] = np.array(run_data["RegionB"][-1])[[1, 2, 3, 5, 6, 7]]
        state[12:18] = np.array(run_data["RegionC"][-1])[[1, 2, 3, 5, 6, 7]]

        if self.rolling:
            times = np.linspace(time, time + self.model_horizon, 101)
        else:
            # times = np.linspace(time, self.model_horizon, 101)
            times = np.arange(time, self.model_horizon, self.delta_t)
        self.model.params['state_init'] = state
        self.model.params['times'] = times

        if np.all(state[1::3] == 0):
            self.current_control = space_model.no_control_policy
            self.next_renew_time += self.update_freq
            self.log.append((time, self.model.run_policy(space_model.no_control_policy), np.nan))
            return

        try:
            cold_start_file = next(self.cold_start_files)
        except StopIteration:
            cold_start_file = None

        time1 = time_mod.time()
        bocop_run = self.model.run_bocop(verbose=False, cold_start=cold_start_file,
                                         init_policy=self.init_policy)
        time_taken = time_mod.time() - time1

        if self.verbose:
            print(bocop_run.exit_code)
        if bocop_run.exit_code != "Optimal Solution Found.":
            warnings.warn("Convergence Failure!")

        self.current_control = bocop_run.control
        self.next_renew_time += self.update_freq

        self.log.append((time, bocop_run, time_taken))

    def get_log(self):
        """Return log of optimiser outputs at update times."""

        return self.log

class SpacePolicyController(MPCSpaceController):
    """Risk structured open loop controller using given policy to allocate resources."""

    def __init__(self, oc_model, policy):
        mpc_params = {
            'update_freq': np.inf,
            'model_horizon': None,
            'rolling_horizon': False,
            'init_policy': None,
            'initial_control_func': policy
        }

        super().__init__(oc_model, mpc_params)
