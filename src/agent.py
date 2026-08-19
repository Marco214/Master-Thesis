"""
Models household decision-making about residential location based on
income, rent burden, green amenities, and social homophily. Agents evaluate
nearby cells and may move if a better location is found or if mobility
rules require it
"""

from mesa import Agent

class Household(Agent):
    def __init__(self, unique_id, model, income_group, income_value):
        super().__init__(unique_id, model)
        self.income_group = income_group  # 'low', 'middle', 'high'
        self.income_value = income_value

        #print("Agent", self.unique_id, self.income_group, self.income_value)

    def utility(self, cell):
        """
        :return a scalar utility combining:
        1) rent burden (rent relative to income and income_group-specific factor),
        2) disposable income utility,
        3) green preference (income-group-weighted * global green_attraction * cell.green_score),
        4) social homophily (fraction of same-income neighbors)
        """
        #1. Rent burden
        demand_factor = cell.occupancy() / max(1, (self.model.width * self.model.height) / 100.0)
        rent = cell.current_rent(demand_factor, self.model.demand_price_elasticity, self.model.beta_ugs)

        rent_burden = rent / (self.income_value *
                              {"low": 0.8, "middle": 1.2, "high": 2.5}[self.income_group])

        #2. Disposable income utility
        income_utility = (self.income_value - rent) / 200.0

        #3. Green preference
        green_weight = {"low": 0.3, "middle": 1.0, "high": 2.5}[self.income_group]
        green_pref = green_weight * self.model.green_attraction * cell.green_score

        #4. Social homophily
        neighbors = self.model.grid.get_neighbors(cell.pos, moore=True, include_center=False, radius=1)
        same = sum(1 for n in neighbors if getattr(n, "income_group", None) == self.income_group)
        social_component = (same / len(neighbors)) * 4.0 if neighbors else 0.0

        #Combine
        util = income_utility + green_pref - rent_burden * 40.0 + social_component
        return util

    def step(self):
        """
        agent's per-tick behavior
        evaluate utilities and move to the best cell if it improves utility
        """
        current_pos = self.pos
        current_cell = self.model.cell_map[current_pos]
        current_util = self.utility(current_cell)

        # High-income always consider moving (higher mobility)
        if self.income_group == "high" or current_util < self.model.utility_threshold:

            #Agent Movement Step
            best_cell = current_cell
            best_util = current_util
            candidates = self.model.sample_cells_around(current_pos, radius=self.model.move_search_radius, k=40)
            for c in candidates:
                u = self.utility(c)
                if u > best_util:
                    best_util = u
                    best_cell = c
            if best_cell is not current_cell:
                try:
                    current_cell.occupants.remove(self)
                except ValueError:
                    pass
                best_cell.occupants.append(self)
                self.model.grid.move_agent(self, best_cell.pos)

