"""Code for handling event rates in simulations"""

import numpy as np
import simulation

class RateSum:
    """Structure to hold and select event rates."""

    def __init__(self, size):
        self.rates = np.zeros(size)
        self.totrate = np.sum(self.rates)
        self.nevents = size

    def insert_rate(self, pos, rate):
        """Insert new rate into position in structure."""

        if rate < 0:
            rate = 0
        # rate_change = rate - self.rates[pos]
        self.rates[pos] = rate
        # self.totrate += rate_change
        self.full_resum()

    def get_rate(self, pos):
        """Extract rate from structure."""
        return self.rates[pos]

    def select_event(self, rate):
        """Randomly choose event from structure, weighted by rates."""
        event_id = 0
        cum_rate = self.rates[0]

        while cum_rate < rate and event_id < self.nevents:
            event_id += 1
            cum_rate += self.rates[event_id]

        return event_id

    def get_total_rate(self):
        """Calculate total rate of all events in structure."""
        return self.totrate

    def full_resum(self):
        """Re-calculate total rate."""
        self.totrate = np.sum(self.rates)

    def zero_rates(self):
        """Set all rates to zero."""
        self.rates = np.zeros(self.nevents)
        self.full_resum()

class VacRateHandler:
    """Structure to store vaccination rates, separated by region for regional control."""

    def __init__(self, nodes, regions=None, region_map=None, node_id_map=None):
        if regions is None:
            self.regions = list(set(node.region for node in nodes))
        else:
            self.regions = regions

        if node_id_map is None or region_map is None:
            counter = {region: 0 for region in self.regions}
            self.region_map = {}
            self.node_id_map = {region: {} for region in self.regions}
            for node in nodes:
                self.node_id_map[node.region][counter[node.region]] = node.id
                self.region_map[node.id] = (node.region, counter[node.region])
                counter[node.region] += 1
        else:
            self.region_map = region_map
            self.node_id_map = node_id_map

        self.region_factors = {region: 1 for region in self.regions}
        self.rates = {region: RateSum(len(self.node_id_map[region])) for region in self.regions}

        self.totrates = {region: self.rates[region].get_total_rate() for region in self.regions}

    def insert_rate(self, pos, rate):
        """Insert new rate into position in structure."""

        region, new_pos = self.region_map[pos]
        self.rates[region].insert_rate(new_pos, rate)
        self.totrates[region] = self.rates[region].get_total_rate()

    def get_rate(self, pos):
        """Extract rate from structure."""
        region, new_pos = self.region_map[pos]
        return self.rates[region].get_rate(new_pos)

    def select_event(self, rate):
        """Randomly choose event from structure, weighted by rates."""
        cumulative_rate = 0.0

        for region in self.regions:
            group_rate = self.totrates[region] * self.region_factors[region]
            if rate < cumulative_rate + group_rate:
                pos = self.rates[region].select_event(
                    (rate - cumulative_rate) / self.region_factors[region])
                return self.node_id_map[region][pos]
            else:
                cumulative_rate += group_rate

    def get_total_rate(self):
        """Calculate total rate of all events in structure."""
        return np.sum([
            self.totrates[region] * self.region_factors[region] for region in self.regions])

    def full_resum(self):
        """Re-calculate total rate."""
        for region in self.regions:
            self.rates[region].full_resum()
            self.totrates[region] = self.rates[region].get_total_rate()

    def zero_rates(self):
        """Set all rates to zero."""
        for region in self.regions:
            self.rates[region].zero_rates()
        self.full_resum()

class RateHandler:
    """Structure to hold all rates, and handle event selection."""

    def __init__(self, nodes, risk_coupling, dist_coupling, params):
        # Set up rate structures

        self.rate_structures = [RateSum(len(nodes)) for _ in simulation.Event]
        self.rate_structures[simulation.Event.VAC_H] = VacRateHandler(nodes)
        self.rate_structures[simulation.Event.VAC_L] = VacRateHandler(
            nodes, regions=self.rate_structures[simulation.Event.VAC_H].regions,
            region_map=self.rate_structures[simulation.Event.VAC_H].region_map,
            node_id_map=self.rate_structures[simulation.Event.VAC_H].node_id_map)

        self.rate_factors = np.array([
            params['birth_rate'], params['death_rate'], 1, params.get('vac_rate', 0),
            params['removal_rate'], params['recov_rate'],
            params['birth_rate'], params['death_rate'], 1, params.get('vac_rate', 0),
            params['removal_rate'], params['recov_rate']
        ])

        self.inf_press = np.zeros((len(simulation.Risk), len(nodes)))
        self.vac_rates = np.ones((len(simulation.Risk), len(nodes)))

        # Initialise rates
        for i, node in enumerate(nodes):
            n_high = node.state[simulation.State.SUS_H] + node.state[simulation.State.INF_H] + node.state[simulation.State.VAC_H]
            n_low = node.state[simulation.State.SUS_L] + node.state[simulation.State.INF_L] + node.state[simulation.State.VAC_L]

            self.rate_structures[simulation.Event.BIRTH_H].insert_rate(i, n_high)
            self.rate_structures[simulation.Event.BIRTH_L].insert_rate(i, n_low)

            self.rate_structures[simulation.Event.DEATH_H].insert_rate(i, n_high)
            self.rate_structures[simulation.Event.DEATH_L].insert_rate(i, n_low)

            for j, node2 in enumerate(nodes):
                self.inf_press[simulation.Risk.HIGH, i] += dist_coupling[i, j] * (
                    risk_coupling[simulation.Risk.HIGH, simulation.Risk.HIGH] * node2.state[simulation.State.INF_H] +
                    risk_coupling[simulation.Risk.HIGH, simulation.Risk.LOW] * node2.state[simulation.State.INF_L])

                self.inf_press[simulation.Risk.LOW, i] += dist_coupling[i, j] * (
                    risk_coupling[simulation.Risk.LOW, simulation.Risk.LOW] * node2.state[simulation.State.INF_L] +
                    risk_coupling[simulation.Risk.LOW, simulation.Risk.HIGH] * node2.state[simulation.State.INF_H])

            self.rate_structures[simulation.Event.INF_H].insert_rate(
                i, self.inf_press[simulation.Risk.HIGH, i] * node.state[simulation.State.SUS_H])
            self.rate_structures[simulation.Event.INF_L].insert_rate(
                i, self.inf_press[simulation.Risk.LOW, i] * node.state[simulation.State.SUS_L])

            self.rate_structures[simulation.Event.REM_H].insert_rate(i, node.state[simulation.State.INF_H])
            self.rate_structures[simulation.Event.REM_L].insert_rate(i, node.state[simulation.State.INF_L])
            self.rate_structures[simulation.Event.REC_H].insert_rate(i, node.state[simulation.State.INF_H])
            self.rate_structures[simulation.Event.REC_L].insert_rate(i, node.state[simulation.State.INF_L])

            self.rate_structures[simulation.Event.VAC_H].insert_rate(
                i, self.vac_rates[simulation.Risk.HIGH, i] * node.state[simulation.State.SUS_H])
            self.rate_structures[simulation.Event.VAC_L].insert_rate(
                i, self.vac_rates[simulation.Risk.LOW, i] * node.state[simulation.State.SUS_L])

    def update_rate_factors(self, new_factors):
        """Update event rate factors."""

        for event, factor in new_factors.items():
            self.rate_factors[event] = factor

    def update_reg_vac_factors(self, new_factors):
        """Update regional vaccination rate factors."""

        for region, risk, factor in new_factors:
            if risk == simulation.Risk.HIGH:
                self.rate_structures[simulation.Event.VAC_H].region_factors[region] = factor
            elif risk == simulation.Risk.LOW:
                self.rate_structures[simulation.Event.VAC_L].region_factors[region] = factor

    def get_total_rate(self):
        """Evaluate total rate across all events."""
        total_rate = np.sum([self.rate_factors[i] *
                             self.rate_structures[i].get_total_rate() for i in simulation.Event])
        return total_rate

    def get_next_event(self):
        """Calculate what event is to happen next."""
        total_rates = [self.rate_factors[i] *
                       self.rate_structures[i].get_total_rate() for i in simulation.Event]
        total_rate = np.sum(total_rates)
        if total_rate < 10e-10:
            return (total_rate, None, None)

        select_rate = np.random.random_sample()*total_rate

        cumulative_rate = 0.0

        for i in simulation.Event:
            group_rate = total_rates[i]
            if select_rate < cumulative_rate + group_rate:
                return (total_rate, i,
                        self.rate_structures[i].select_event(
                            (select_rate - cumulative_rate)/self.rate_factors[i]))
            else:
                cumulative_rate += group_rate

        return (total_rate, None, None)

    def get_rate(self, node_id, event, with_rate_factor=True):
        """Extract current rate value from structures."""

        if with_rate_factor:
            multiplier = self.rate_factors[event]
        else:
            multiplier = 1

        return multiplier * self.rate_structures[event].get_rate(node_id)

    def adjust_rate(self, node_id, value, event, rate_change=True):
        """Update rate stored in rate structure."""

        if rate_change:
            old_rate = self.rate_structures[event].get_rate(node_id)
            new_rate = old_rate + value
        else:
            new_rate = value

        self.rate_structures[event].insert_rate(node_id, new_rate)

    def adjust_inf_press(self, node_id, value, risk, rate_change=True):
        """Update stored infectious pressure."""

        if rate_change:
            old_pres = self.inf_press[risk, node_id]
            new_pres = old_pres + value
        else:
            new_pres = value

        self.inf_press[risk, node_id] = new_pres

        return new_pres

    def adjust_vac_rate(self, node_id, value, risk, rate_change=True):
        """Update stored vaccine pressure."""

        if rate_change:
            old_rate = self.vac_rates[risk, node_id]
            new_rate = old_rate + value
        else:
            new_rate = value

        self.vac_rates[risk, node_id] = new_rate

        return new_rate
