"""Code for running network based simulation model using direct Gillespie algorithm."""

import warnings
from collections import namedtuple
from operator import add
import copy
from enum import IntEnum
import numpy as np
import rate_handling

##################
## Define Enums ##
##################

class State(IntEnum):
    """Host states."""
    SUS_H = 0
    INF_H = 1
    VAC_H = 2
    REM_H = 3
    SUS_L = 4
    INF_L = 5
    VAC_L = 6
    REM_L = 7

class Risk(IntEnum):
    """Host risk classes."""
    HIGH = 0
    LOW = 1

class Event(IntEnum):
    """Possible event types."""
    BIRTH_H = 0
    DEATH_H = 1
    INF_H = 2
    VAC_H = 3
    REM_H = 4
    REC_H = 5
    BIRTH_L = 6
    DEATH_L = 7
    INF_L = 8
    VAC_L = 9
    REM_L = 10
    REC_L = 11

HIGH_LIVE_STATES = {State.SUS_H, State.INF_H, State.VAC_H}
LOW_LIVE_STATES = {State.SUS_L, State.INF_L, State.VAC_L}

# Define state changes for each event
# Event: (Before, After)
EVENT_DEFNS = {
    Event.BIRTH_H: (None, State.SUS_H),
    Event.INF_H: (State.SUS_H, State.INF_H),
    Event.VAC_H: (State.SUS_H, State.VAC_H),
    Event.REM_H: (State.INF_H, State.REM_H),
    Event.REC_H: (State.INF_H, State.SUS_H),
    Event.BIRTH_L: (None, State.SUS_L),
    Event.INF_L: (State.SUS_L, State.INF_L),
    Event.VAC_L: (State.SUS_L, State.VAC_L),
    Event.REM_L: (State.INF_L, State.REM_L),
    Event.REC_L: (State.INF_L, State.SUS_L)
}

SimulationRun = namedtuple("SimulationRun", ["run_data", "event_data", "objective", "control"])
SimulationObj = namedtuple("SimulationObj", ["epidemic_cost", "control_cost"])

####################
## Define Classes ##
####################

class Node:
    """Class to store node information."""

    def __init__(self, position, region, state, id_num):
        self.position = np.array(position)
        self.region = region
        self.state = state
        self.id = id_num

    def __eq__(self, other):
        if isinstance(other, Node):
            return (np.all(self.position == other.position) and self.region == other.region and
                    self.state == other.state and self.id == other.id)
        return False

class Simulator:
    """Main class to create and run simulations.

    Initialisation requires:
        nodes_init:     List of n Node objects used to initialise epidemic *or* filename to
                        initialise from (see simulation.initialise_nodes)
        risk_coupling:  2x2 array giving coupling between risk groups
        dist_coupling:  nxn array giving couplings between nodes
        params:         Dictionary of additional parameters (see below)
        controller:     Controller object from control_tools module for vaccination control

    Additional parameters and default values in params dict:
        'end_time':                     Stop simulations at this time (required)
        'return_event_data':            If True return info on all state changes (default: False)
        'controller_args':              Dict of any arguments to pass to controller object
                                            (default: {})
        'control_update_freq':          How often to reoptimise controller (default: np.inf)
        'update_control_on_all_events': Whether to reallocate control after every state change
                                            (default: False)

    """

    def __init__(self, nodes_init, risk_coupling, dist_coupling, params, controller=None):
        if isinstance(nodes_init, str):
            # Initialise from file
            self.nodes = initialise_nodes(nodes_init)
        elif isinstance(nodes_init, list) and np.all([isinstance(node, Node)
                                                      for node in nodes_init]):
            # Initialise from list
            self.nodes = nodes_init
        else:
            raise TypeError("Unknown type for node initialisation!")

        self.risk_coupling = np.array(risk_coupling)
        if self.risk_coupling.shape != (2, 2):
            raise ValueError("Risk coupling array must be 2x2!!")

        self.dist_coupling = np.array(dist_coupling)
        if self.dist_coupling.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("Distance coupling shape doesn't match nodes!!")

        self.params = params
        self.system = None

        self.controller = controller

    def run_simulation(self, nruns=1, verbose=True):
        """Carry out Gillespie run.

        Return list of run results, or single run results if nruns=1
        """

        results = []

        for i in range(nruns):
            self.system = System(copy.deepcopy(self.nodes), self.risk_coupling, self.dist_coupling,
                                 self.params, self.controller)

            next_time, next_event, event_id = self.system.get_next_event()

            while next_event is not None and next_time <= self.params['end_time']:
                self.system.carry_out_event(next_time, next_event, event_id)
                next_time, next_event, event_id = self.system.get_next_event()

            # Calculate objective value integral I dt
            total_inf = np.array([x[1+State.INF_H] + x[1+State.INF_L]
                                  for x in self.system.run_data['Global']])
            times = np.array([x[0] for x in self.system.run_data['Global'][1:]] +
                             [self.params['end_time']])
            for j in reversed(range(len(times) - 1)):
                times[j+1] -= times[j]
            objective = np.sum(total_inf * times)

            if self.controller is not None:
                if hasattr(self.controller, "get_objective"):
                    objective = SimulationObj(objective, self.system.controller.get_objective())
                else:
                    warnings.warn("Controller has no get_objective function. Objectives will not "
                                  "include control costs!")
                    objective = SimulationObj(objective, 0)
            else:
                objective = SimulationObj(objective, 0)

            if hasattr(self.system, "all_events"):
                event_data = self.system.all_events
            else:
                event_data = None

            if self.controller is not None:
                try:
                    control_data = self.system.controller.get_log()
                except AttributeError:
                    control_data = None
            else:
                control_data = None

            results.append(SimulationRun(self.system.run_data, event_data, objective, control_data))
            if verbose:
                print("Run {0} of {1} complete.".format(i+1, nruns))

        if nruns == 1:
            results = results[0]

        return results

def initialise_nodes(node_file):
    """Read node information from file.

    File format is one row per node with the following space delimited values:
        node id (numbered 0 to n-1)
        x position
        y position
        region name
        initial state: S_H I_H V_H R_H S_L I_L V_L R_L
    """

    nodes = []
    id_check = 0

    with open(node_file, "r") as infile:
        lines = [line.strip() for line in infile]
        lines = [line for line in lines if line]
        for line in lines:
            node_id, xpos, ypos, region, *state = line.split()
            if int(node_id) != id_check:
                raise ValueError("Wrong node id order!!")
            state = [int(x) for x in state]
            nodes.append(Node((float(xpos), float(ypos)), region, state, int(node_id)))
            id_check += 1

    return nodes

class System:
    """Class to hold system level info and event handling for simulations.

    Initialisation same as for simulation.Simulator
    """

    def __init__(self, nodes, risk_coupling, dist_coupling, params, controller):
        self.nodes = nodes
        self.risk_coupling = risk_coupling
        self.dist_coupling = dist_coupling
        self.params = params

        self.rate_handler = rate_handling.RateHandler(nodes, risk_coupling, dist_coupling, params)
        self.run_data = {}
        self.initialise_run_data()

        self.new_events = None
        self.time = 0.0

        if self.params.get("return_event_data", False):
            self.all_events = []

        if controller is not None:
            control_args = self.params.get("controller_args", {})
            self.controller = controller(**control_args)
            self.update_control()
            if self.params.get("control_update_freq", None) is not None:
                self.next_intervention_update = self.params.get("control_update_freq", np.inf)
            else:
                self.next_intervention_update = np.inf
        else:
            self.controller = None
            self.next_intervention_update = np.inf

    def initialise_run_data(self):
        """Setup data structure to store simulation results.

        run_data object stores times and state globally, for nodes, and for regions.
        """

        self.run_data["Global"] = [[0.0] + [0]*8]

        for node in self.nodes:
            self.run_data["Node" + str(node.id)] = [[0.0] + node.state]
            try:
                self.run_data["Region" + str(node.region)][0] = list(map(
                    add, [0.0] + node.state, self.run_data["Region" + str(node.region)][0]))
            except KeyError:
                self.run_data["Region" + str(node.region)] = [[0.0] + node.state]
            self.run_data["Global"][0] = list(map(
                add, [0.0] + node.state, self.run_data["Global"][0]))

    def update_control(self, scheduled=False):
        """Run controller to update vaccination rates.

        scheduled is used to specify a reoptimisation.
        """

        new_rates, new_events, new_factors, new_reg_vac_factors = self.controller(
            self.time, self.nodes, self.run_data, self.new_events, self.rate_handler.vac_rates,
            scheduled=scheduled)

        for node_id, old_state, new_state in new_events:
            node = self.nodes[node_id]
            node.state[old_state] -= 1
            node.state[new_state] += 1
            self.update_states(node, old_state, new_state)

        for node_id, risk, rate in new_rates:
            self.rate_handler.adjust_vac_rate(node_id, rate, risk, rate_change=False)
            if risk == Risk.HIGH:
                self.rate_handler.adjust_rate(node_id, rate * self.nodes[node_id][State.SUS_H],
                                              Event.VAC_H, rate_change=False)
            elif risk == Risk.LOW:
                self.rate_handler.adjust_rate(node_id, rate * self.nodes[node_id][State.SUS_L],
                                              Event.VAC_L, rate_change=False)
            else:
                raise ValueError("Unknown Risk value!!")

        self.rate_handler.update_rate_factors(new_factors)
        self.rate_handler.update_reg_vac_factors(new_reg_vac_factors)

    def carry_out_event(self, time, event, event_id):
        """Carry out event on node, updating states and rates."""

        node = self.nodes[event_id]

        if time < self.time:
            raise ValueError("Event happening before current time!")

        # if death event risk group mustbe chosen randomly
        if event == Event.DEATH_H:
            norm = np.sum([node.state[x] for x in HIGH_LIVE_STATES])
            probabilities = [node.state[x]/norm for x in HIGH_LIVE_STATES]
            old_state = np.random.choice(
                list(HIGH_LIVE_STATES), p=probabilities)
            new_state = None

        elif event == Event.DEATH_L:
            norm = np.sum([node.state[x] for x in LOW_LIVE_STATES])
            probabilities = [node.state[x]/norm for x in LOW_LIVE_STATES]
            old_state = np.random.choice(
                list(LOW_LIVE_STATES), p=probabilities)
            new_state = None

        else:
            old_state, new_state = EVENT_DEFNS[event]

        if old_state is not None:
            if node.state[old_state] <= 0:
                raise ValueError("Cannot carry out event! No {0} hosts in node {1}!".format(
                    old_state.name, node.id))
            node.state[old_state] -= 1

        if new_state is not None:
            node.state[new_state] += 1

        self.time = time
        self.update_states(node, old_state, new_state)

        if self.new_events is not None:
            self.new_events.append((time, event_id, old_state, new_state))
        else:
            self.new_events = [(time, event_id, old_state, new_state)]

        if self.params.get("return_event_data", False):
            self.all_events.append(((time, event_id, old_state, new_state)))

        if self.controller is not None and self.params.get('update_control_on_all_events', False):
            self.update_control()

    def distribute_inf_change(self, node_id, increase=True, risk=Risk.HIGH):
        """Distribute change in infection rates to all nodes, default increase rates."""

        # whether to increase or decrease rates
        if increase:
            multiplier = +1
        else:
            multiplier = -1

        for adjust_node in self.nodes:
            if self.dist_coupling[adjust_node.id, node_id] > 0:
                self.rate_handler.inf_press[Risk.HIGH, adjust_node.id] += multiplier * (
                    self.risk_coupling[Risk.HIGH, risk] *
                    self.dist_coupling[adjust_node.id, node_id])

                self.rate_handler.inf_press[Risk.LOW, adjust_node.id] += multiplier * (
                    self.risk_coupling[Risk.LOW, risk] *
                    self.dist_coupling[adjust_node.id, node_id])

                self.rate_handler.adjust_rate(
                    adjust_node.id, self.rate_handler.inf_press[Risk.HIGH, adjust_node.id] *
                    adjust_node.state[State.SUS_H], Event.INF_H, rate_change=False)
                self.rate_handler.adjust_rate(
                    adjust_node.id, self.rate_handler.inf_press[Risk.LOW, adjust_node.id] *
                    adjust_node.state[State.SUS_L], Event.INF_L, rate_change=False)

    def update_states(self, node, old_state, new_state):
        """Store event info and carry out rate updates."""

        if old_state is None and new_state is None:
            raise ValueError("No event to carry out!")

        self.run_data["Node" + str(node.id)].append([self.time] + node.state)

        self.run_data["Region" + str(node.region)].append(
            self.run_data["Region" + str(node.region)][-1].copy())
        if new_state is not None:
            self.run_data["Region" + str(node.region)][-1][1 + new_state] += 1
        if old_state is not None:
            self.run_data["Region" + str(node.region)][-1][1 + old_state] -= 1
        self.run_data["Region" + str(node.region)][-1][0] = self.time

        self.run_data["Global"].append(self.run_data["Global"][-1].copy())
        if new_state is not None:
            self.run_data["Global"][-1][1 + new_state] += 1
        if old_state is not None:
            self.run_data["Global"][-1][1 + old_state] -= 1
        self.run_data["Global"][-1][0] = self.time

        self.distribute_state_change(node, old_state, new_state)

    def distribute_state_change(self, node, old_state, new_state):
        """Distribute rate changes after state change."""

        if old_state in HIGH_LIVE_STATES and new_state not in HIGH_LIVE_STATES:
            # Decrease high risk birth rate
            self.rate_handler.adjust_rate(node.id, -1, Event.BIRTH_H)
            # Decrease high risk death rate
            self.rate_handler.adjust_rate(node.id, -1, Event.DEATH_H)
        elif old_state not in HIGH_LIVE_STATES and new_state in HIGH_LIVE_STATES:
            # Increase high risk birth rate
            self.rate_handler.adjust_rate(node.id, +1, Event.BIRTH_H)
            # Increase high risk death rate
            self.rate_handler.adjust_rate(node.id, +1, Event.DEATH_H)
        elif old_state in LOW_LIVE_STATES and new_state not in LOW_LIVE_STATES:
            # Decrease low risk birth rate
            self.rate_handler.adjust_rate(node.id, -1, Event.BIRTH_L)
            # Decrease low risk death rate
            self.rate_handler.adjust_rate(node.id, -1, Event.DEATH_L)
        elif old_state not in LOW_LIVE_STATES and new_state in LOW_LIVE_STATES:
            # Increase low risk birth rate
            self.rate_handler.adjust_rate(node.id, +1, Event.BIRTH_L)
            # Increase low risk death rate
            self.rate_handler.adjust_rate(node.id, +1, Event.DEATH_L)

        if old_state == State.SUS_H:
            # Decrease infection rate
            new_rate = node.state[State.SUS_H] * self.rate_handler.inf_press[Risk.HIGH, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.INF_H, rate_change=False)
            # Decrease vaccination rate
            new_rate = node.state[State.SUS_H] * self.rate_handler.vac_rates[Risk.HIGH, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.VAC_H, rate_change=False)
        elif old_state == State.INF_H:
            # Distribute infection rate decrease
            self.distribute_inf_change(node.id, increase=False, risk=Risk.HIGH)
            # Decrease high risk removal and recovery rates
            self.rate_handler.adjust_rate(node.id, -1, Event.REM_H)
            self.rate_handler.adjust_rate(node.id, -1, Event.REC_H)
        elif old_state == State.SUS_L:
            # Decrease infection rate
            new_rate = node.state[State.SUS_L] * self.rate_handler.inf_press[Risk.LOW, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.INF_L, rate_change=False)
            # Decrease vaccination rate
            new_rate = node.state[State.SUS_L] * self.rate_handler.vac_rates[Risk.LOW, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.VAC_L, rate_change=False)
        elif old_state == State.INF_L:
            # Distribute infection rate decrease
            self.distribute_inf_change(node.id, increase=False, risk=Risk.LOW)
            # Decrease low risk removal and recovery rates
            self.rate_handler.adjust_rate(node.id, -1, Event.REM_L)
            self.rate_handler.adjust_rate(node.id, -1, Event.REC_L)

        if new_state == State.SUS_H:
            # Increase infection rate
            new_rate = node.state[State.SUS_H] * self.rate_handler.inf_press[Risk.HIGH, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.INF_H, rate_change=False)
            # Increase vaccination rate
            new_rate = node.state[State.SUS_H] * self.rate_handler.vac_rates[Risk.HIGH, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.VAC_H, rate_change=False)
        elif new_state == State.INF_H:
            # Distribute infection rate increase
            self.distribute_inf_change(node.id, increase=True, risk=Risk.HIGH)
            # Increase high risk removal and recovery rates
            self.rate_handler.adjust_rate(node.id, +1, Event.REM_H)
            self.rate_handler.adjust_rate(node.id, +1, Event.REC_H)
        elif new_state == State.SUS_L:
            # Increase infection rate
            new_rate = node.state[State.SUS_L] * self.rate_handler.inf_press[Risk.LOW, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.INF_L, rate_change=False)
            # Increase vaccination rate
            new_rate = node.state[State.SUS_L] * self.rate_handler.vac_rates[Risk.LOW, node.id]
            self.rate_handler.adjust_rate(node.id, new_rate, Event.VAC_L, rate_change=False)
        elif new_state == State.INF_L:
            # Distribute infection rate increase
            self.distribute_inf_change(node.id, increase=True, risk=Risk.LOW)
            # Increase low risk removal and recovery rates
            self.rate_handler.adjust_rate(node.id, +1, Event.REM_L)
            self.rate_handler.adjust_rate(node.id, +1, Event.REC_L)

    def get_next_event(self):
        """Calculate next event and time to carry out."""

        total_rate, next_event, event_id = self.rate_handler.get_next_event()

        if next_event is not None:
            next_time = self.time + (-1.0/total_rate)*np.log(np.random.random_sample())
        else:
            next_time = np.inf

        while next_time > self.next_intervention_update:
            if self.next_intervention_update >= self.params['end_time']:
                break

            self.time = self.next_intervention_update
            self.next_intervention_update += self.params["control_update_freq"]
            self.update_control(scheduled=True)

            total_rate, next_event, event_id = self.rate_handler.get_next_event()

            if next_event is not None:
                next_time = self.time + (-1.0/total_rate)*np.log(np.random.random_sample())
            else:
                next_time = np.inf

        return next_time, next_event, event_id
