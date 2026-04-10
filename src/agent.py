from mesa import Agent

class Household(Agent):
    def __init__(self, unique_id, model, income_group, income_value):
        super().__init__(unique_id, model)
        self.income_group = income_group  # 'low', 'middle', 'high'
        self.income_value = income_value

        #print("Agent", self.unique_id, self.income_group, self.income_value)

    def utility(self, cell):
        """
        Utility function combining:
        - disposable income after rent (income_value - rent)
        - attractiveness from green_score (MCDA aggregated)
        - social preference: prefer neighbors of similar income
        """
        demand_factor = cell.occupancy / max(1, (self.model.width * self.model.height) / 100.0)
        rent = cell.current_rent(demand_factor, self.model.demand_price_elasticity, self.model.beta_ugs)
        rent_burden = rent / (self.income_value + 1e-6)
        income_utility = (self.income_value - rent) / 100.0
        green_pref = self.model.green_attraction * cell.green_score
        # social component: fraction of neighbors of same group within Moore neighborhood (agents)
        same_count = 0
        neigh_count = 0
        neighbors = self.model.grid.get_neighbors(cell.pos, moore=True, include_center=False, radius=1)
        for other in neighbors:
            neigh_count += 1
            if getattr(other, "income_group", None) == self.income_group:
                same_count += 1
        social_component = (same_count / neigh_count) if neigh_count > 0 else 0.0
        group_sensitivity = {"low": 1.6, "middle": 1.0, "high": 0.6}[self.income_group]
        util = income_utility * (1.0 / group_sensitivity) + green_pref - rent_burden * 50.0 + social_component * 5.0
        return util

    def step(self):
        current_pos = self.pos
        current_cell = self.model.cell_map[current_pos]
        current_util = self.utility(current_cell)

        if current_util < self.model.utility_threshold:
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

                print("Agent", self.unique_id, self.income_group, self.income_value, "moved to ", best_cell.pos)

