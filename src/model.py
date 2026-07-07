"""
Gentrification model extended with Urban Green Spaces (UGS)
- Parks with attributes: proximity, size, quality, function
- MCDA aggregation to compute cell.green_score (Caprioli-style)
- Hedonic capitalization of green_score into rents (Bottero/Rigolon references)
- Heatmap export with park overlay and safe folder clearing
"""

import os
import random
import math
import numpy as np
import pandas as pd

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

from agent import Household

# -------------------------
# Parameter Defaults
# -------------------------
DEFAULTS = {
    # grid parameter (values from Mauro)
    "width": 9,
    "height": 9,
    "n_agents": 1000,
    # park parameters
    "park_coverage": 0.12, # fraction of cells that become park centroids
    "park_size_small": 1000,
    "park_size_medium": 10000,
    "park_size_large": 100000,
    # parameters for rent calculation from https://data.census.gov/table/ACSDT1Y2022.B25064?q=median+gross+rent&y=2022
    "base_rent_mean": 7.2, # Median income of 1300$
    "base_rent_sigma": 0.5,
    "demand_price_elasticity": 0.08,
    # agents preferences
    "green_attraction": 50.0,
    "move_search_radius": 4,
    "utility_threshold": 20,
    "seed": None,
    # heatmap export params
    "export_every": 3,
    "out_dir": "../output/heatmaps",
    # MCDA weights for UGS attributes (from literature)
    "w_proximity": 0.40,
    "w_size": 0.10,
    "w_quality": 0.25,
    "w_function": 0.25,
    # Hedonic capitalization parameter (beta_ugs)
    "beta_ugs": 0.2,
}

# -------------------------
# Helper functions
# -------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def pick_random_row(df, model, percent_cumul_limit_low = 0, percent_cumul_limit_high = 100):
    df1 = df[(df["percent_cumul"] >= percent_cumul_limit_low) & (df["percent_cumul"] <= percent_cumul_limit_high)]
    total_percent = df1["percent"].sum()
    df1.loc[:, "percent"] = df1["percent"] / total_percent
    return model.random.choices(df1.index, weights = df1["percent"])[0]

# -------------------------
# UGS (park) class
# Park attributes: proximity, size, quality, function
# -------------------------
class UGS:
    def __init__(self, pos, size_m2, quality, function):
        self.pos = pos
        self.size = size_m2
        self.quality = clamp(quality, 0.0, 1.0)   # 0..1
        self.function = function                        #recreation, sports, greenway

# -------------------------
# Cell object stored in cell_map
# -------------------------
class Cell:
    def __init__(self, pos, base_rent, green_score=0.0):
        self.pos = pos
        self.base_rent = base_rent
        self.green_score = green_score
        self.occupants = []

    @property
    def occupancy(self):
        return len(self.occupants)

    def current_rent(self, demand_factor, demand_price_elasticity, beta_ugs):
        """
        Rent formation: base_rent adjusted by demand and green capitalization (hedonic effect).
        """
        rent = self.base_rent * (1.0 + demand_factor * demand_price_elasticity)
        rent *= (1.0 + beta_ugs * self.green_score)
        return rent

# -------------------------
# Main Green-Gentrification Model
# -------------------------
class GreenGentModel(Model):
    def __init__(self, **kwargs):
        params = DEFAULTS.copy()
        params.update(kwargs)
        if params["seed"] is not None:
            random.seed(params["seed"])
            np.random.seed(params["seed"])

        # model size and grid
        self.width = params["width"]
        self.height = params["height"]
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)
        self.running = True

        # parameters
        self.green_attraction = params["green_attraction"]
        self.demand_price_elasticity = params["demand_price_elasticity"]
        self.utility_threshold = params["utility_threshold"]
        self.move_search_radius = params["move_search_radius"]
        self.beta_ugs = params.get("beta_ugs", DEFAULTS["beta_ugs"])

        # UGS attribute weights (MCDA)
        self.w_prox = params.get("w_proximity", DEFAULTS["w_proximity"])
        self.w_size = params.get("w_size", DEFAULTS["w_size"])
        self.w_quality = params.get("w_quality", DEFAULTS["w_quality"])
        self.w_function = params.get("w_function", DEFAULTS["w_function"])

        # heatmap export settings
        self.export_every = params.get("export_every", DEFAULTS["export_every"])
        self.out_dir = params.get("out_dir", DEFAULTS["out_dir"])
        ensure_dir(self.out_dir)

        # create cell map
        self.cell_map = {}
        for x in range(self.width):
            for y in range(self.height):
                #base = max(50.0, random.gauss(params["base_rent_mean"], params["base_rent_sd"]))
                base = np.clip(np.random.lognormal(params["base_rent_mean"], params["base_rent_sigma"]), 750.0, 5000.0)
                # initial green_score placeholder (will be computed from parks)
                cell = Cell((x, y), base, green_score=0.0)
                self.cell_map[(x, y)] = cell

        # create parks (UGS) as centroids with attributes
        self.parks = []
        n_parks = max(1, int(self.width * self.height * params.get("park_coverage", DEFAULTS["park_coverage"])))
        for _ in range(n_parks):
            px = random.randrange(self.width)
            py = random.randrange(self.height)
            size = random.choice([DEFAULTS["park_size_small"], DEFAULTS["park_size_medium"],
                                  DEFAULTS["park_size_large"]])  # sizes in m2 for small, medium and large parks
            quality = clamp(random.gauss(0.7, 0.30))
            function = random.choice(["recreation", "sports", "greenway"])
            self.parks.append(UGS((px, py), size, quality, function))

        # spawn agents, generate incomes
        self.df = pd.read_csv("../income/income_clean.csv") # share of income_group over all agents (values from https://www.ssa.gov/cgi-bin/netcomp.cgi?year=2022)
        n = params["n_agents"]
        agent_id = 0
        group = ""
        all_positions = list(self.cell_map.keys())
        for i in range(n):
            row = pick_random_row(df=self.df, model=self, percent_cumul_limit_high=99.99911)
            if row <= 5:  # 0 - 29.999$ per year (38%)
                group = "low"
            elif row > 5 and row <= 35: # 30.000$ - 174.999$ per year (57%)
                group = "middle"
            elif row > 35: # >175.000$ per year (5%)
                group = "high"

            income = self.random.uniform(self.df["bound_low"][row], self.df["bound_high"][row]) / 12 # calculate per month
            a = Household(agent_id, self, group, income)
            pos = random.choice(all_positions)
            self.grid.place_agent(a, pos)
            self.cell_map[pos].occupants.append(a)
            self.schedule.add(a)
            agent_id += 1

        # compute initial green scores from parks (MCDA)
        self.compute_green_scores()

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "low_count": lambda m: m.count_group("low"),
                "middle_count": lambda m: m.count_group("middle"),
                "high_count": lambda m: m.count_group("high"),
                "avg_rent": lambda m: m.average_rent(),
                "avg_green": lambda m: m.average_green(),
                "high_local": lambda m: m.local_high_income_share(),
            }
        )

        # collect initial state so datacollector is not empty
        self.datacollector.collect(self)

    # -------------------------
    # MCDA + proximity: compute green_score per cell
    # -------------------------
    def compute_green_scores(self, decay_scale=None):
        """
        For each cell compute an aggregated UGS influence score using:
        - proximity (distance decay)
        - park size (normalized)
        - park quality (0..1)
        - park function match (simple categorical weight)
        We use the maximum park influence per cell (closest/best park) as in Caprioli-style approach.
        """
        if decay_scale is None:
            decay_scale = max(self.width, self.height) / 4.0

        for (x, y), cell in self.cell_map.items():
            best_score = 0.0
            for park in self.parks:
                # Euclidean distance in grid units
                dist = math.hypot(park.pos[0] - x, park.pos[1] - y)
                # proximity: exponential decay
                prox = math.exp(-dist / decay_scale)
                # size normalization (log scale to compress large parks)
                size_norm = math.log(park.size + 1) / math.log(10000 + 1)
                # function match: simple example weights
                # influence on gentrification: greenway > recreation > sport
                func_weight = {"recreation": 0.4, "sports": 0.1, "greenway": 0.5}.get(park.function, 0.3)
                # MCDA weighted sum
                score = (self.w_prox * prox +
                         self.w_size * size_norm +
                         self.w_quality * park.quality +
                         self.w_function * func_weight)
                if score > best_score:
                    best_score = score
            cell.green_score = clamp(best_score, 0.0, 1.0)

    # -------------------------
    # Utility helpers
    # -------------------------
    def count_group(self, group):
        return sum(1 for a in self.schedule.agents if a.income_group == group)

    def average_rent(self):
        rents = []
        for cell in self.cell_map.values():
            demand_factor = cell.occupancy / max(1, (self.width * self.height) / 100.0)
            rents.append(cell.current_rent(demand_factor, self.demand_price_elasticity, self.beta_ugs))
        return float(np.mean(rents))

    def average_green(self):
        return float(np.mean([c.green_score for c in self.cell_map.values()]))

    def local_high_income_share(self, radius=3):
        """
        Durchschnittlicher High-Income-Anteil um alle Parks.
        """
        if len(self.parks) == 0:
            return 0.0

        shares = []

        for park in self.parks:
            px, py = park.pos

            high = 0
            total = 0

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):

                    x = px + dx
                    y = py + dy

                    if 0 <= x < self.width and 0 <= y < self.height:

                        cell = self.cell_map[(x, y)]

                        for a in cell.occupants:
                            total += 1
                            if a.income_group == "high":
                                high += 1

            if total > 0:
                shares.append(high / total)

        if len(shares) == 0:
            return 0.0

        return np.mean(shares)

    def sample_cells_around(self, pos, radius=5, k=30):
        x0, y0 = pos
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    candidates.append(self.cell_map[(x, y)])
        if len(candidates) <= k:
            return candidates
        return random.sample(candidates, k)

    # -------------------------
    # Heatmap export utilities (with park overlay)
    # -------------------------
    def build_income_grid(self):
        grid = np.full((self.width, self.height), np.nan)
        for (x, y), cell in self.cell_map.items():
            if cell.occupancy > 0:
                incomes = [a.income_value for a in cell.occupants]
                grid[x, y] = np.mean(incomes)
            else:
                grid[x, y] = np.nan
        return grid

    def build_high_income_share_grid(self):
        """
        Gibt ein 2D-Array zurück, das pro Zelle den Anteil der High-Income-Haushalte enthält.
        Wertebereich: 0.0 bis 1.0
        """
        grid = np.zeros((self.width, self.height))

        for (x, y), cell in self.cell_map.items():
            if cell.occupancy == 0:
                grid[x, y] = np.nan  # leere Zellen als NaN anzeigen
                continue

            high = sum(1 for a in cell.occupants if a.income_group == "high")
            total = cell.occupancy

            grid[x, y] = high / total

        return grid

    def build_rent_grid(self):
        grid = np.zeros((self.width, self.height))
        for (x, y), cell in self.cell_map.items():
            demand_factor = cell.occupancy / max(1, (self.width * self.height) / 100.0)
            grid[x, y] = cell.current_rent(demand_factor, self.demand_price_elasticity, self.beta_ugs)
        return grid

    def build_green_grid(self):
        grid = np.zeros((self.width, self.height))
        for (x, y), cell in self.cell_map.items():
            grid[x, y] = cell.green_score
        return grid

    def save_heatmap(self, array2d, title, label, filename, cmap, vmin, vmax, overlay_parks):
        plt.figure(figsize=(6, 6))
        img = np.transpose(array2d)

        # prepare park marker scaling (log scale)
        sizes = [getattr(p, "size", 2000) for p in self.parks] if len(self.parks) else [1000]
        log_sizes = [math.log10(max(1, s)) for s in sizes]
        min_log, max_log = min(log_sizes), max(log_sizes)
        min_marker, max_marker = 80, 1200

        ax = plt.gca()

        # draw mesh first (cells)
        X = np.arange(img.shape[0] + 1)
        Y = np.arange(img.shape[1] + 1)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color=(0.9, 0.9, 0.9))
        mesh = ax.pcolormesh(X, Y, img, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                             edgecolors='white', linewidth=0.6, shading='auto')
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(label, fontsize=12)

        # overlay park centroids and size/quality markers ON TOP of the mesh
        if overlay_parks and len(self.parks) > 0:
            for park in self.parks:
                px, py = park.pos
                # center markers in the cell for pcolormesh
                cx, cy = px + 0.5, py + 0.5

                # compute marker size on log scale
                log_s = math.log10(max(1, park.size))
                if max_log == min_log:
                    size_marker = (min_marker + max_marker) / 2
                else:
                    size_marker = min_marker + (log_s - min_log) * (max_marker - min_marker) / (max_log - min_log)
                size_marker = float(size_marker)

                ax.scatter(cx, cy, c='lime', s=size_marker, edgecolors='k',
                           linewidths=0.5, marker='o', zorder=3)
                ax.text(cx + 0.2, cy + 0.2, f"q={park.quality:.2f}\nf={park.function}",
                        color='white', fontsize=6, zorder=4,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    def export_heatmaps(self, step):
        income_grid = self.build_income_grid()
        rent_grid = self.build_rent_grid()
        high_share_grid = self.build_high_income_share_grid()

        if np.isnan(income_grid).all():
            income_vmin, income_vmax = 0, 1
        else:
            income_vmin = np.nanpercentile(income_grid, 5)
            income_vmax = np.nanpercentile(income_grid, 95)
        rent_vmin, rent_vmax = np.percentile(rent_grid, 5), np.percentile(rent_grid, 95)

        fname_income = os.path.join(self.out_dir, f"income_year_{step:02d}.png")
        fname_rent_parks = os.path.join(self.out_dir, f"rent_year_{step:02d}.png")

        fname_high_share = os.path.join(self.out_dir, f"high_income_share_year_{step:02d}.png")

        self.save_heatmap(income_grid, f"Average Income (Year {step})", f"Income per month in $", fname_income,
                          cmap="cividis", vmin=income_vmin, vmax=income_vmax, overlay_parks=True)
        self.save_heatmap(rent_grid, f"Rent with Parks (Year {step})", f"Rent per month in $", fname_rent_parks,
                          cmap="inferno", vmin=rent_vmin, vmax=rent_vmax, overlay_parks=True)
        self.save_heatmap(
            high_share_grid,
            f"High-Income Share (Year {step})",
            "Share of high-income households",
            fname_high_share,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            overlay_parks=True
        )

    # -------------------------
    # Step: agents act, then rents update via demand feedback
    # 1 step equals 1 year
    # -------------------------
    def step(self):
        self.schedule.step()
        for cell in self.cell_map.values():
            demand_factor = cell.occupancy / max(1, (self.width * self.height) / 100.0)
            # base rent drift influenced by occupancy and green_score
            drift = 1.0 + 0.003 * math.log1p(demand_factor) + 0.002 * cell.green_score
            cell.base_rent *= drift

            #print(f"Cell {cell.pos}-> base_rent:{round(cell.base_rent, 1)}, Green-Score:{round(cell.green_score, 2)}")

        # collect data
        self.datacollector.collect(self)

        # debug logging every N steps
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        if self._step_count % DEFAULTS["export_every"] == 0 or self._step_count == 1:
            base_rents = [c.base_rent for c in self.cell_map.values()]
            demand_rents = [
                c.current_rent(c.occupancy / max(1, (self.width * self.height) / 100.0),
                               self.demand_price_elasticity,
                               self.beta_ugs)
                for c in self.cell_map.values()
            ]

            income_grid = self.build_income_grid()  # returns np.array with NaNs for empty cells
            valid = ~np.isnan(income_grid)
            if np.any(valid):
                vals = income_grid[valid]
                inc_min = float(np.min(vals))
                inc_max = float(np.max(vals))
                inc_median = float(np.median(vals))
                #print(
                    #f"[CELL INCOME] Step {self._step_count}: min={inc_min:.2f}  median={inc_median:.2f}  max={inc_max:.2f}")
            #else:
                #print(f"[CELL INCOME] Step {self._step_count}: no occupied cells")

            #print(f"[CELL RENT] Step {self._step_count}: base_rent min={min(base_rents):.2f} max={max(base_rents):.2f} "
                  #f"current_rent min={min(demand_rents):.2f} max={max(demand_rents):.2f}")

    # -------------------------
    # Run with heatmap export
    # -------------------------
    def run_model(self, steps):
        for step in range(steps):
            self.step()
            if (step % self.export_every) == 0:
                self.export_heatmaps(step)