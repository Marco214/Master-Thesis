# Green-Gentrification Agent-Based Model

An agent-based model (ABM), built with [Mesa](https://mesa.readthedocs.io/), that simulates how urban green spaces (UGS) — parks — influence residential sorting, rent formation, and gentrification dynamics across an income-stratified population of households.

The model explores a well-documented urban phenomenon: the installation or improvement of parks tends to raise nearby property values (a "green amenity" or hedonic capitalization effect), which can price out lower-income residents and shift neighborhood composition toward higher-income groups over time ("green gentrification").

## Project structure

| File | Role |
|---|---|
| `model.py` | Defines the Mesa `Model` (`GreenGentModel`), the grid of cells, the parks (UGS), rent formation, green-score computation, cost accounting, data collection, and heatmap visualization/export. |
| `agent.py` | Defines the `Household` agent, its utility function, and its per-step decision of whether/where to move. |
| `simulation.py` | Example entry point: configures and runs a single simulation for an example city, exports heatmaps over time, and plots aggregate income/rent trends. |
| `experiments.py` | Batch experiment runner: sweeps park parameters (size, quality, proximity, function, and optionally cost variants) across many simulation runs to detect **gentrification tipping points**, in parallel if desired, and produces aggregated results and heatmaps. |

## Core concepts

### The grid and cells (`model.py`)
The city is represented as a rectangular grid (`width` × `height`) of `Cell` objects. Each cell has:
- a **base rent**, initialized from a lognormal distribution (calibrated loosely to US Census median gross rent data),
- a **green score** (0–1), reflecting the influence of nearby parks,
- a list of **occupants** (households currently living there).

Rent at a cell (`current_rent`) is the base rent adjusted upward by local demand pressure (`demand_price_elasticity`) and by the **hedonic capitalization** of its green score (`beta_ugs`): cells near attractive, high-quality parks become more expensive.

### Parks / Urban Green Spaces (`UGS` class in `model.py`)
A configurable number of parks (`park_coverage`) are randomly placed on the grid, each with:
- a **size** (small/medium/large, in m²),
- a **quality** (0–1, drawn from a truncated normal distribution),
- a **function** (`recreation`, `sports`, or `greenway`).

Each park's **green_score contribution** to nearby cells is computed via a Multi-Criteria Decision Analysis (MCDA) weighted sum of:
- **proximity** (exponential distance decay),
- **size** (log-normalized, so huge parks don't dominate linearly),
- **quality**,
- **function weight** (greenways score highest, then recreation, then sports).

A cell's final `green_score` is the *maximum* influence among all parks (i.e., the best/closest park dominates), clamped to [0, 1].

Optionally (`enable_park_costs`), parks also carry **investment** and **annual operating costs** based on size and quality, which can be allocated back to nearby cells/tenants as a rent surcharge (`apply_costs_to_rents`), using the same proximity-decay logic.

### Households (`agent.py`)
Each `Household` agent belongs to an income group (`low`, `middle`, `high`) with an associated monthly `income_value`, sampled at model initialization from an empirical income-share distribution (loaded from a CSV: `../income/income_clean.csv`).

At every step, each household evaluates its **current cell's utility** and, if:
- it belongs to the `high` income group (assumed to be more mobile), **or**
- its current utility falls below a global `utility_threshold`,

it searches a random sample of nearby cells (`sample_cells_around`, within `move_search_radius`) and relocates to the best-utility alternative found, if any beats its current cell.

#### Utility function
For a household evaluating a cell, utility combines four weighted components:
1. **Rent burden** — rent relative to income, scaled by an income-group-specific affordability factor (lower-income groups are penalized more heavily for the same rent burden).
2. **Disposable income** — remaining income after rent, scaled down.
3. **Green preference** — income-group-weighted attraction to the cell's green score (higher-income households value green space more).
4. **Social homophily** — the share of Moore-neighborhood neighbors sharing the same income group (households prefer to live near similar-income peers).

These combine as:
```
utility = disposable_income − 40 × rent_burden + green_preference + social_homophily
```

### Model dynamics (`GreenGentModel.step`, `model.py`)
Each simulated step (interpreted as one year):
1. All households act (Mesa's `RandomActivation` schedule) — evaluating and potentially moving.
2. Every cell's base rent drifts upward slightly based on local occupancy demand and its green score (a slow, systemic gentrification pressure independent of individual moves).
3. If cost accounting is enabled, total park investment/operational costs are recomputed and (optionally) capitalized into rents.
4. Aggregate statistics are collected via Mesa's `DataCollector`.

### Data collection & outputs
The `DataCollector` tracks, per step:
- household counts by income group (`low_count`, `middle_count`, `high_count`),
- average rent and average green score across the grid,
- the average share of high-income households living near parks (`local_high_income_share`),
- total park investment and annual operating costs (if cost accounting is enabled).

`export_heatmaps` (called periodically via `export_every`) renders and saves PNG heatmaps for:
- **average income per cell** (`cividis` colormap),
- **rent** (`inferno` colormap),
- **high-income household share** (`Reds` colormap),

each with park locations overlaid as scaled markers (size ∝ park size, log scale) annotated with quality and function.

## Running the simulation for an example city (`simulation.py`)

`run_example(steps, seed)`:
1. Prepares (and clears) the output directory `../output/heatmaps`.
2. Instantiates `GreenGentModel` with a 7×7 grid, 1000 households, and specific MCDA/rent-capitalization parameters.
3. Exports the initial state and a standalone green-score reference map.
4. Runs the model for the given number of steps, exporting heatmaps periodically and collecting average income/rent/green-score time series.
5. Plots average income and average rent over time using Matplotlib.

To run directly:
```bash
python simulation.py
```
By default this runs 50 steps with `seed=122`.

## Batch experiments & tipping-point analysis (`experiments.py`)

While `simulation.py` runs a single illustrative city, `experiments.py` is designed for **systematic parameter sweeps** across many independent simulation runs, in order to identify **tipping points**: the conditions under which a park triggers a persistent, self-reinforcing shift toward higher-income occupancy nearby.

### Tipping-point detection (`detect_tipping_local`)
For a given run, a "tipping point" (Kipppunkt) is defined as the first year `t` at which **all** of the following hold:
1. **Rent threshold**: the model-wide average rent has risen to at least `rent_rel_threshold` × the baseline (initial) average rent (default: +10%).
2. **Persistence**: this elevated rent level is sustained for at least `persist_years` consecutive years (default: 3) — a temporary spike doesn't count.
3. **Local social shift**: the share of high-income households living near the reference park (`high_local`, from the `DataCollector`) has increased by at least `income_shift_threshold` (default: 0.003) relative to the start of the run.

If all three conditions are met, the run is flagged as tipped (`kipp = True`) along with the year it occurred (`kipp_time`).

### Single-run wrapper (`run_single_experiment`)
Runs one full simulation for a given parameter combination:
- Instantiates `GreenGentModel` with deterministic seeding.
- Optionally **overrides the park configuration** entirely — replacing the model's randomly generated parks with a single controlled park at a given position, size, quality, function, and proximity decay scale (`decay_scale`), and randomly re-shuffling all households across the grid so the experiment starts from a clean, symmetric baseline.
- If cost accounting is enabled on the model, computes the new park's investment/operational costs consistently with the model's cost parameters.
- Runs the model for `steps` years and returns the resulting time series, whether/when a tipping point occurred, the computed rent threshold, the rent level at the tipping point, and total investment/operational costs.
- Can optionally export the full time series to CSV (`export=True`).

### Parallelized grid sweep (`run_parameter_grid` / `_worker_task`)
Orchestrates a **4-dimensional grid sweep** over:
- `size_values` — park sizes to test,
- `quality_values` — park quality levels,
- `proximity_values` — proximity decay scales,
- `function_values` — park functions (`recreation`, `sports`, `greenway`),

with `n_runs` repeated seeds per combination for statistical robustness. If `model_base_kwargs` enables park costs (`enable_park_costs=True`), the sweep instead runs a **fixed set of 4 cost scenarios** (combinations of size, quality, investment cost/m², and operating cost/m², at a fixed proximity and function) rather than the full geometric grid.

Key features:
- **Multiprocessing** (`use_multiprocessing=True`): distributes runs across CPU cores using a `multiprocessing.Pool` with a `spawn` context; falls back to sequential execution otherwise.
- **Incremental progress reporting**: as soon as all repeated runs for a given parameter combination complete, prints the tipping probability (`p_kipp`) and median tipping year for that combination.
- **Incremental checkpointing**: periodically (every ~5% of jobs) writes partial aggregated results to `grid_results_partial.csv`, so long sweeps can be monitored or recovered from mid-run.
- **Error resilience**: any run that raises an exception is captured with its error message rather than crashing the whole sweep, and failed runs are collected in `grid_errors.csv`.
- **Final aggregation**: for each parameter combination, computes the tipping probability (`p_kipp`), median tipping year, rent threshold, rent level at tipping, and (if applicable) rounded investment/operational costs, saved to `grid_results.csv`. A `run_info.txt` file records total runtime and job count.

### Visualization helpers
- `plot_heatmap_from_grid`: generic 2D pivot-table heatmap plotter for any two swept parameters against a result metric.
- Inside `example_run`, a dedicated cost-scenario heatmap helper (`save_cost_heatmap`) plots **tipping probability** and **median tipping year** as a function of total investment cost vs. total annual operational cost, for the 4 predefined cost scenarios.

### Example sweep (`example_run`)
Runs a default 4×4×3×3 parameter grid (park size × quality × proximity × function) with `n_runs=1` and `steps=50` on a 7×7 grid with 1000 households, `enable_park_costs=False`, saving results to `../output/experiments`. If cost-related columns are present in the results, it also produces the two cost-scenario heatmaps described above.

To run directly:
```bash
python experiments.py
```

**Note:** running the full example grid sweep involves many simulation runs (potentially dozens to hundreds, depending on `n_runs` and multiprocessing settings) and can take significantly longer than the single-city example in `simulation.py`.

## Requirements
- Python 3
- [`mesa`](https://pypi.org/project/Mesa/) (Version 2.1.1 note: uses the older `mesa.time.RandomActivation` / `mesa.space.MultiGrid` API)
- `numpy`
- `pandas`
- `networkX`
- `matplotlib`
- `multiprocessing` (standard library; used by `experiments.py` for parallel grid sweeps)

## Data dependencies
`model.py` expects an income distribution CSV at `../income/income_clean.csv`, containing (at minimum) columns `percent`, `percent_cumul`, `bound_low`, and `bound_high`, used to probabilistically assign each household's income group and monthly income.

## Known parameters worth tuning
| Parameter | Effect |
|---|---|
| `beta_ugs` | Strength of green-space hedonic capitalization into rent |
| `green_attraction` | Global scaling of how much green score matters to utility |
| `w_proximity` / `w_size` / `w_quality` / `w_function` | MCDA weights for how a park's attributes translate into green score |
| `utility_threshold` | How dissatisfied a (non-high-income) household must be before it searches for a new home |
| `move_search_radius` / `k` (in `sample_cells_around`) | Household search behavior when relocating |
| `park_coverage` | Density of parks across the grid |
| `enable_park_costs` / `apply_costs_to_rents` | Whether park investment/operating costs feed back into rents |
| `rent_rel_threshold` / `income_shift_threshold` / `persist_years` (in `experiments.py`) | Thresholds defining what counts as a gentrification "tipping point" |

## Notes / caveats
- `experiments.py` mutates a model's parks and household placement in-place when overriding park parameters; this is intended for controlled single-park tipping-point experiments and is not equivalent to the fully randomized multi-park setup used in `simulation.py`.
