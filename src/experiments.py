"""
Grid-sweep experiments for tipping-point analysis of the gentrification of urban green spaces
"""
import os
import time
import random
import multiprocessing
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import shutil

from model import GreenGentModel

@dataclass
class ExperimentConfig:
    """Central configuration for all experiment parameters."""
    size_values: list
    quality_values: list
    proximity_values: list
    function_values: list
    n_runs: int = 10
    steps: int = 50
    rent_rel_threshold: float = 1.10
    income_shift_threshold: float = 0.003
    persist_years: int = 3
    out_dir: str = "../output/experiments"
    model_kwargs = {
        'width': 7,
        'height': 7,
        'n_agents': 1000,
        'enable_park_costs': True
    }
    base_seed: int = 42
    n_workers: int | None = None
    use_multiprocessing: bool = True

# Helper functions
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def clear_dir(path):
    """
    Entfernt alle Dateien und Unterordner in `path`, lässt aber den Ordner selbst bestehen.
    Wenn der Ordner nicht existiert, wird er angelegt.
    """
    ensure_dir(path)
    # Entferne alle Inhalte sicher
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        try:
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
        except Exception as e:
            # Robustheit: Fehler protokollieren, aber nicht abbrechen
            print(f"Warnung: konnte {full_path} nicht löschen: {e}")


def safe_remove_from_list(lst, item):
    try:
        lst.remove(item)
    except ValueError:
        pass

def detect_tipping_local(model, baseline_rent,
                         park_pos=None,
                         rent_rel_threshold=ExperimentConfig.rent_rel_threshold,
                         income_shift_threshold=ExperimentConfig.income_shift_threshold,
                         persist_years=ExperimentConfig.persist_years,):
    """
    Tipping-point analysis based on the local proportion of high-income households in a park's neighborhood
    """
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    if df.empty:
        return False, None

    if park_pos is None:
        if not getattr(model, "parks", None):
            park_pos = None
        else:
            park_pos = model.parks[0].pos

    if 'avg_rent' not in df.columns:
        return False, None
    if 'high_local' not in df.columns:
        share_high_local = [0.0] * len(df)
    else:
        share_high_local = df['high_local'].tolist()

    baseline_rent_val = baseline_rent

    # Tipping Point Analysis
    for t in range(1, len(df)):
        rent_t = df['avg_rent'].iloc[t]

        # 1) Rent-treshold
        if rent_t >= baseline_rent_val * rent_rel_threshold:

            end_idx = min(len(df), t + int(persist_years))
            # 2) Persistence
            if (df['avg_rent'].iloc[t:end_idx] >= baseline_rent_val * rent_rel_threshold).all():

                # 3) local social change
                high_change = share_high_local[t] - share_high_local[0] if len(share_high_local) > t else 0.0
                if high_change >= income_shift_threshold:
                    return True, t

    return False, None

def write_run_statistics(df_grid, output_file= f"{ExperimentConfig.out_dir}/results/run_info.txt"):
    """
    Writes overall statistics of the median tipping times and optionally p_kipp
    over all parameter combinations to run_info.txt.

    Parameters
    ----------
    df_grid : Any

    output_file : str
        Path to run_info.txt
    """
    # --- median_kipp_time statistics ---
    median_values = df_grid["median_kipp_time"].to_numpy(dtype=float)
    median_values = median_values[~np.isnan(median_values)]
    n = len(median_values)

    with open(output_file, "a") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("OVERALL MEDIAN KIPP TIME STATISTICS\n")
        f.write("=" * 60 + "\n")

        if n == 0:
            f.write("No valid median_kipp_time values available.\n")
        else:
            mean = float(np.mean(median_values))
            median = float(np.median(median_values))
            std = float(np.std(median_values, ddof=1)) if n > 1 else 0.0
            minimum = float(np.min(median_values))
            maximum = float(np.max(median_values))
            q25 = float(np.percentile(median_values, 25))
            q75 = float(np.percentile(median_values, 75))
            iqr = q75 - q25

            if n > 1 and std > 0.0:
                try:
                    ci_low, ci_high = t.interval(confidence=0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
                except Exception:
                    ci_low, ci_high = mean, mean
            else:
                ci_low, ci_high = mean, mean

            f.write(f"Number of parameter combinations : {n}\n\n")
            f.write(f"Mean                           : {mean:.3f}\n")
            f.write(f"Median                         : {median:.3f}\n")
            f.write(f"Standard deviation             : {std:.3f}\n\n")
            f.write(f"Minimum                        : {minimum:.3f}\n")
            f.write(f"25 % Quantile                  : {q25:.3f}\n")
            f.write(f"75 % Quantile                  : {q75:.3f}\n")
            f.write(f"Maximum                        : {maximum:.3f}\n")
            f.write(f"Interquartile Range (IQR)      : {iqr:.3f}\n\n")
            f.write(f"95 % Confidence Interval       : [{ci_low:.3f}, {ci_high:.3f}]\n")

    print(f"Median kipp time statistics written to {output_file}")

    # --- p_kipp statistics ---
    p_values = df_grid["p_kipp"].to_numpy(dtype=float)
    p_values = p_values[~np.isnan(p_values)]
    n_p = len(p_values)

    with open(output_file, "a") as f:
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("OVERALL P_KIPP STATISTICS\n")
        f.write("=" * 60 + "\n")

        if n_p == 0:
            f.write("No valid p_kipp values available.\n")
        else:
            p_mean = float(np.mean(p_values))
            p_median = float(np.median(p_values))
            p_std = float(np.std(p_values, ddof=1)) if n_p > 1 else 0.0
            p_min = float(np.min(p_values))
            p_max = float(np.max(p_values))
            p_q25 = float(np.percentile(p_values, 25))
            p_q75 = float(np.percentile(p_values, 75))
            p_iqr = p_q75 - p_q25

            if n_p > 1 and p_std > 0.0:
                try:
                    p_ci_low, p_ci_high = t.interval(confidence=0.95, df=n_p - 1, loc=p_mean, scale=p_std / np.sqrt(n_p))
                except Exception:
                    p_ci_low, p_ci_high = p_mean, p_mean
            else:
                p_ci_low, p_ci_high = p_mean, p_mean

            f.write(f"Mean                           : {p_mean:.3f}\n")
            f.write(f"Median                         : {p_median:.3f}\n")
            f.write(f"Standard deviation             : {p_std:.3f}\n\n")
            f.write(f"Minimum                        : {p_min:.3f}\n")
            f.write(f"25 % Quantile                  : {p_q25:.3f}\n")
            f.write(f"75 % Quantile                  : {p_q75:.3f}\n")
            f.write(f"Maximum                        : {p_max:.3f}\n")
            f.write(f"Interquartile Range (IQR)      : {p_iqr:.3f}\n\n")
            f.write(f"95 % Confidence Interval       : [{p_ci_low:.3f}, {p_ci_high:.3f}]\n")

        f.write("=" * 60 + "\n")

    print(f"p_kipp statistics written to {output_file}")


def run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years, export=False):
    """
    Single-run wrapper
    params: dict with keys that will be passed to GreenGentModel (e.g., park_size, park_quality, decay_scale, park_function)
    seed: int
    returns: dict with run result (kipp, kipp_time, df, investment_cost, annual_operational_cost)
    """
    # Deterministic seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    model_base_kwargs = params.get('model_base_kwargs', {})
    model_kwargs = model_base_kwargs.copy()
    model_kwargs['seed'] = seed
    model = GreenGentModel(**model_kwargs)

    # Park-Override
    if 'park_size' in params or 'park_quality' in params or 'park_pos' in params or 'decay_scale' in params or 'park_function' in params:
        try:
            from model import UGS as UGS_local
        except Exception:
            UGS_local = None

        model.parks = []
        px, py = params.get('park_pos', (model.width // 2, model.height // 2))
        size = params.get('park_size', 10000)
        quality = params.get('park_quality', 0.7)
        function = params.get('park_function', random.choice(["recreation", "sports", "greenway"]))

        ugs_cls = getattr(model, 'UGS', None) or UGS_local
        if ugs_cls is None:
            class _UGS_fallback:
                def __init__(self, pos, size_m2, quality, function):
                    self.pos = pos
                    self.size = size_m2
                    self.quality = quality
                    self.function = function
                    self.investment_cost = 0.0
                    self.annual_operational_cost = 0.0
            ugs_cls = _UGS_fallback

        # append park
        new_park = ugs_cls((px, py), size, quality, function)
        model.parks.append(new_park)

        # If the model has capitalized costs, calculate the costs for the new park
        if getattr(model, "enable_park_costs", False):
            invest_per_m2 = getattr(model, "cost_invest_per_m2", None)
            op_per_m2 = getattr(model, "cost_operational_per_m2_per_year", None)
            q_inv_mult = getattr(model, "quality_invest_multiplier", 0.0)
            q_op_mult = getattr(model, "quality_operational_multiplier", 0.0)

            if hasattr(new_park, "compute_costs") and invest_per_m2 is not None and op_per_m2 is not None:
                new_park.compute_costs(invest_per_m2, op_per_m2, q_inv_mult, q_op_mult)
            else:
                new_park.investment_cost = new_park.size * (invest_per_m2 if invest_per_m2 is not None else 0.0) * (
                            1.0 + new_park.quality * q_inv_mult)
                new_park.annual_operational_cost = new_park.size * (op_per_m2 if op_per_m2 is not None else 0.0) * (
                            1.0 + new_park.quality * q_op_mult)

            # Update aggregated totals in the model
            if hasattr(model, "compute_investment_cost"):
                model.investment_cost = model.compute_investment_cost()
            else:
                model.investment_cost = float(sum(getattr(p, "investment_cost", 0.0) for p in model.parks))

            if hasattr(model, "compute_annual_operational_cost"):
                model.annual_operational_cost = model.compute_annual_operational_cost()
            else:
                model.annual_operational_cost = float(
                    sum(getattr(p, "annual_operational_cost", 0.0) for p in model.parks))

        decay = params.get('decay_scale', None)
        if hasattr(model, 'compute_green_scores'):
            model.compute_green_scores(decay_scale=decay)

        for a in list(model.schedule.agents):
            old_pos = getattr(a, 'pos', None)
            if old_pos is None:
                continue
            occupants = model.cell_map.get(old_pos).occupants if old_pos in model.cell_map else []
            if a in occupants:
                safe_remove_from_list(occupants, a)

        # new random positions
        all_positions = list(model.cell_map.keys())
        for a in list(model.schedule.agents):
            pos = random.choice(all_positions)
            try:
                model.grid.place_agent(a, pos)
            except Exception:
                a.pos = pos
            model.cell_map[pos].occupants.append(a)

    # run model
    model.run_model(steps)

    # collect time series
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)

    # Read the total cost from the model (if available)
    investment_cost = getattr(model, "investment_cost", None)
    annual_operational_cost = getattr(model, "annual_operational_cost", None)

    if df.empty or 'avg_rent' not in df.columns:
        return {'kipp': False, 'kipp_time': None, 'df': df,
                'investment_cost': investment_cost,
                'annual_operational_cost': annual_operational_cost}
    baseline_rent = df['avg_rent'].iloc[0]

    kipp, kipp_time = detect_tipping_local(
        model,
        baseline_rent=baseline_rent,
        park_pos=params.get('park_pos', model.parks[0].pos if model.parks else None),
        rent_rel_threshold=rent_rel_threshold,
        income_shift_threshold=income_shift_threshold,
        persist_years=persist_years
    )

    # Calculate the rent threshold and rental value at the tipping point (if applicable)
    rent_threshold = None
    rent_at_kipp = None
    try:
        rent_threshold = float(baseline_rent * rent_rel_threshold)
    except Exception:
        rent_threshold = None

    if kipp and kipp_time is not None and 'avg_rent' in df.columns and len(df) > kipp_time:
        try:
            rent_at_kipp = float(df['avg_rent'].iloc[int(kipp_time)])
        except Exception:
            rent_at_kipp = None

    if export:
        fname = os.path.join(params.get('out_dir', 'results'),
                             f"run_size{params.get('park_size')}_qual{params.get('park_quality')}_prox{params.get('decay_scale')}_func{params.get('park_function')}_seed{seed}.csv")
        ensure_dir(os.path.dirname(fname))
        df.to_csv(fname, index=False)

    return {
        'kipp': kipp,
        'kipp_time': kipp_time,
        'df': df,
        'rent_threshold': rent_threshold,
        'rent_at_kipp': rent_at_kipp,
        'investment_cost': investment_cost,
        'annual_operational_cost': annual_operational_cost
    }

def _worker_task(task):
    """
    Worker wrapper for multiprocessing (must be top-level)
    task: tuple containing (size, quality, proximity, park_function, run_idx, seed, model_base_kwargs, steps, rent_rel_threshold, income_shift_threshold, persist_years, out_dir)
    returns: dict with keys describing the run and result
    """
    (size, quality, proximity, park_function, run_idx, seed,
     model_base_kwargs, steps, rent_rel_threshold, income_shift_threshold, persist_years, out_dir) = task

    params = {
        'park_size': size,
        'park_quality': quality,
        'decay_scale': proximity,
        'park_function': park_function,
        'model_base_kwargs': model_base_kwargs or {},
        'out_dir': out_dir
    }
    try:
        res = run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years,
                                    export=False)

        total_inv = res.get('investment_cost', None)
        total_op = res.get('annual_operational_cost', None)

        try:
            total_inv_out = int(round(float(total_inv))) if total_inv is not None else None
        except Exception:
            total_inv_out = None

        try:
            total_op_out = int(round(float(total_op))) if total_op is not None else None
        except Exception:
            total_op_out = None

        return {
            'park_size': size,
            'park_quality': quality,
            'decay_scale': proximity,
            'park_function': park_function,
            'run_idx': run_idx,
            'seed': seed,
            'kipp': res['kipp'],
            'kipp_time': res['kipp_time'],
            'rent_threshold': res.get('rent_threshold'),
            'rent_at_kipp': res.get('rent_at_kipp'),
            'investment_cost': total_inv_out,
            'annual_operational_cost': total_op_out
        }

    except Exception as e:
        # If an error occurs, return a robust structure
        return {
            'park_size': size,
            'park_quality': quality,
            'decay_scale': proximity,
            'park_function': park_function,
            'run_idx': run_idx,
            'seed': seed,
            'kipp': False,
            'kipp_time': None,
            'error': str(e),
            'investment_cost': None,
            'annual_operational_cost': None
        }

def run_parameter_grid(config: ExperimentConfig):
    """Run grid sweep using a single ExperimentConfig object."""

    size_values = config.size_values
    quality_values = config.quality_values
    proximity_values = config.proximity_values
    function_values = config.function_values
    n_runs = config.n_runs
    steps = config.steps
    rent_rel_threshold = config.rent_rel_threshold
    income_shift_threshold = config.income_shift_threshold
    persist_years = config.persist_years
    out_dir = f"{config.out_dir}/results"
    model_base_kwargs = config.model_kwargs or {}
    base_seed = config.base_seed
    n_workers = config.n_workers
    use_multiprocessing = config.use_multiprocessing
    """
    Grid sweep orchestrator with multiprocessing
    Runs a 4D grid sweep. Set use_multiprocessing=False to run sequentially
    """

    # --- Neu: Vor jedem Grid-Sweep die Zielordner leeren ---
    heatmaps_dir = os.path.join(config.out_dir, "experiment_heatmaps")
    results_dir = out_dir  # already points to config.out_dir/results

    # Stelle sicher, dass die Ordner existieren und leere sie
    clear_dir(heatmaps_dir)
    clear_dir(results_dir)
    ensure_dir(out_dir)

    results = []
    start_time = time.perf_counter()

    tasks = []
    job_counter = 0

    enable_costs = False
    if model_base_kwargs and isinstance(model_base_kwargs, dict):
        enable_costs = bool(model_base_kwargs.get('enable_park_costs', False))

    if enable_costs:
        fixed_proximity = 2.0
        fixed_function = "recreation"
        combos = [
            (100000, 0.8, 80.0, 2.5),
            (100000, 0.4, 80.0, 1.0),
            (10000, 0.8, 40.0, 2.5),
            (10000, 0.4, 40.0, 1.0),
        ]
        for (size, quality, invest_cost, op_cost) in combos:
            for run_idx in range(n_runs):
                seed = int(base_seed) + job_counter + run_idx
                mbk = (model_base_kwargs.copy() if model_base_kwargs is not None else {}).copy()
                mbk['enable_park_costs'] = True
                mbk['cost_invest_per_m2'] = invest_cost
                mbk['cost_operational_per_m2_per_year'] = op_cost
                task = (size, quality, fixed_proximity, fixed_function, run_idx, seed,
                        mbk, steps, rent_rel_threshold, income_shift_threshold, persist_years, out_dir)
                tasks.append(task)
            job_counter += n_runs
    else:
        for size in size_values:
            for quality in quality_values:
                for proximity in proximity_values:
                    for park_function in function_values:
                        for run_idx in range(n_runs):
                            seed = int(base_seed) + job_counter + run_idx
                            task = (size, quality, proximity, park_function, run_idx, seed,
                                    model_base_kwargs or {}, steps, rent_rel_threshold, income_shift_threshold,
                                    persist_years, out_dir)
                            tasks.append(task)
                        job_counter += n_runs

    total_jobs = len(tasks)
    if total_jobs == 0:
        return pd.DataFrame(results)

    cpu_count = multiprocessing.cpu_count() or 1
    if n_workers is None:
        n_workers = max(1, min(cpu_count, total_jobs))
    else:
        n_workers = max(1, min(n_workers, cpu_count, total_jobs))

    raw_results = []
    save_every = max(1, total_jobs // 20)
    completed_counts = {}
    printed_combos = set()

    # Sequential Mode
    if not use_multiprocessing:
        for i, task in enumerate(tasks):
            res = _worker_task(task)
            raw_results.append(res)

            key = (res['park_size'], res['park_quality'], res['decay_scale'], res['park_function'])
            completed_counts[key] = completed_counts.get(key, 0) + 1

            if completed_counts[key] >= n_runs and key not in printed_combos:
                combo_group = [r for r in raw_results
                               if (r['park_size'], r['park_quality'], r['decay_scale'], r['park_function']) == key]
                kipp_count = sum(1 for r in combo_group if r.get('kipp') == True)
                n_runs_actual = len(combo_group)
                kipp_times = [float(r['kipp_time']) for r in combo_group if
                              r.get('kipp') == True and r.get('kipp_time') is not None]
                p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                median_kipp_time = int(np.median(kipp_times)) if len(kipp_times) > 0 else None
                print(f"[RESULT] size={key[0]:>7}  quality={key[1]:.2f}  decay={key[2]:.2f}  func={key[3]:>9}  "
                      f"p_kipp={p_kipp:.3f}  median_kipp_time={median_kipp_time}  runs={n_runs_actual}")
                printed_combos.add(key)

            if (i + 1) % save_every == 0 or (i + 1) == total_jobs:
                df_partial = pd.DataFrame(raw_results)
                grouped = df_partial.groupby(['park_size', 'park_quality', 'decay_scale', 'park_function'])
                partial_rows = []
                for name, group in grouped:
                    size, quality, proximity, park_function = name
                    kipp_count = int(group['kipp'].sum())
                    n_runs_actual = len(group)
                    kipp_times = group.loc[group['kipp'] == True, 'kipp_time'].dropna().astype(float).tolist()
                    p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                    median_kipp_time = (float(np.median(kipp_times)) if len(kipp_times) > 0 else None)
                    rent_threshold_vals = group['rent_threshold'].dropna().astype(float).tolist() if 'rent_threshold' in group else []
                    rent_threshold = float(rent_threshold_vals[0]) if len(rent_threshold_vals) > 0 else None
                    rent_at_kipp_vals = group['rent_at_kipp'].dropna().astype(float).tolist() if 'rent_at_kipp' in group else []
                    rent_at_kipp = float(np.median(rent_at_kipp_vals)) if len(rent_at_kipp_vals) > 0 else None
                    p_kipp_out = round(p_kipp, 2)
                    median_kipp_time_out = (round(median_kipp_time, 2) if median_kipp_time is not None else np.nan)
                    rent_threshold_out = (round(rent_threshold, 2) if rent_threshold is not None else np.nan)
                    rent_at_kipp_out = (round(rent_at_kipp, 2) if rent_at_kipp is not None else np.nan)
                    invest_vals = group['investment_cost'].dropna().astype(float).tolist() if 'investment_cost' in group else []
                    op_vals = group['annual_operational_cost'].dropna().astype(float).tolist() if 'annual_operational_cost' in group else []
                    invest_cost_out = int(round(invest_vals[0])) if len(invest_vals) > 0 else np.nan
                    op_cost_out = int(round(op_vals[0])) if len(op_vals) > 0 else np.nan
                    partial_rows.append({
                            'size': size,
                            'quality': quality,
                            'proximity': proximity,
                            'function': park_function,
                            'investment_cost': invest_cost_out,
                            'annual_operational_cost': op_cost_out,
                            'p_kipp': p_kipp_out,
                            'median_kipp_time': median_kipp_time_out,
                            'rent_threshold': rent_threshold_out,
                            'rent_at_kipp': rent_at_kipp_out,
                            'n_runs': n_runs_actual
                    })
                df_partial_out = pd.DataFrame(partial_rows)
                tmp_path = os.path.join(out_dir, "grid_results_partial.tmp.csv")
                df_partial_out.to_csv(tmp_path, index=False)
                os.replace(tmp_path, os.path.join(out_dir, "grid_results_partial.csv"))

    # Multiprocessing Mode
    else:
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=n_workers)
        chunksize = max(1, total_jobs // (n_workers * 4))
        try:
            for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize)):
                raw_results.append(res)
                key = (res['park_size'], res['park_quality'], res['decay_scale'], res['park_function'])
                completed_counts[key] = completed_counts.get(key, 0) + 1
                if completed_counts[key] >= n_runs and key not in printed_combos:
                    combo_group = [r for r in raw_results
                                   if (r['park_size'], r['park_quality'], r['decay_scale'], r['park_function']) == key]
                    kipp_count = sum(1 for r in combo_group if r.get('kipp') == True)
                    n_runs_actual = len(combo_group)
                    kipp_times = [float(r['kipp_time']) for r in combo_group if
                                  r.get('kipp') == True and r.get('kipp_time') is not None]
                    p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                    median_kipp_time = int(np.median(kipp_times)) if len(kipp_times) > 0 else None
                    print(f"[RESULT] size={key[0]:>7}  quality={key[1]:.2f}  decay={key[2]:.2f}  func={key[3]:>9}  "
                          f"p_kipp={p_kipp:.3f}  median_kipp_time={median_kipp_time}  runs={n_runs_actual}")
                    printed_combos.add(key)
                if (i + 1) % save_every == 0 or (i + 1) == total_jobs:
                    df_partial = pd.DataFrame(raw_results)
                    grouped = df_partial.groupby(['park_size', 'park_quality', 'decay_scale', 'park_function'])
                    partial_rows = []
                    for name, group in grouped:
                        size, quality, proximity, park_function = name
                        kipp_count = int(group['kipp'].sum())
                        n_runs_actual = len(group)
                        kipp_times = group.loc[group['kipp'] == True, 'kipp_time'].dropna().astype(float).tolist()
                        p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                        median_kipp_time = (float(np.median(kipp_times)) if len(kipp_times) > 0 else None)
                        rent_threshold_vals = group['rent_threshold'].dropna().astype(float).tolist() if 'rent_threshold' in group else []
                        rent_threshold = float(rent_threshold_vals[0]) if len(rent_threshold_vals) > 0 else None
                        rent_at_kipp_vals = group['rent_at_kipp'].dropna().astype(float).tolist() if 'rent_at_kipp' in group else []
                        rent_at_kipp = float(np.median(rent_at_kipp_vals)) if len(rent_at_kipp_vals) > 0 else None
                        p_kipp_out = round(p_kipp, 2)
                        median_kipp_time_out = (round(median_kipp_time, 2) if median_kipp_time is not None else np.nan)
                        rent_threshold_out = (round(rent_threshold, 2) if rent_threshold is not None else np.nan)
                        rent_at_kipp_out = (round(rent_at_kipp, 2) if rent_at_kipp is not None else np.nan)
                        invest_vals = group['investment_cost'].dropna().astype(float).tolist() if 'investment_cost' in group else []
                        op_vals = group['annual_operational_cost'].dropna().astype(float).tolist() if 'annual_operational_cost' in group else []
                        invest_cost_out = int(round(invest_vals[0])) if len(invest_vals) > 0 else np.nan
                        op_cost_out = int(round(op_vals[0])) if len(op_vals) > 0 else np.nan
                        partial_rows.append({
                                'size': size,
                                'quality': quality,
                                'proximity': proximity,
                                'function': park_function,
                                'investment_cost': invest_cost_out,
                                'annual_operational_cost': op_cost_out,
                                'p_kipp': p_kipp_out,
                                'median_kipp_time': median_kipp_time_out,
                                'rent_threshold': rent_threshold_out,
                                'rent_at_kipp': rent_at_kipp_out,
                                'n_runs': n_runs_actual
                        })
                    df_partial_out = pd.DataFrame(partial_rows)
                    tmp_path = os.path.join(out_dir, "grid_results_partial.tmp.csv")
                    df_partial_out.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, os.path.join(out_dir, "grid_results_partial.csv"))
        finally:
            pool.close()
            pool.join()

    # Final aggregation
    df_runs = pd.DataFrame(raw_results)
    if df_runs.empty:
        # No runs completed, return empty DataFrame
        return pd.DataFrame(results)

    # Save any errors
    if 'error' in df_runs.columns:
        err_df = df_runs[df_runs['error'].notnull()]
        if not err_df.empty:
            err_df.to_csv(os.path.join(out_dir, "grid_errors.csv"), index=False)

    grouped = df_runs.groupby(['park_size', 'park_quality', 'decay_scale', 'park_function'])
    for name, group in grouped:
        size, quality, proximity, park_function = name

        # number of runs and kipp count
        kipp_count = int(group['kipp'].sum())
        n_runs_actual = len(group)

        # collect kipp times (only where kipp==True and kipp_time not null)
        kipp_times = group.loc[(group['kipp'] == True) & (group['kipp_time'].notnull()), 'kipp_time'].astype(
            float).tolist()

        # p_kipp and median_kipp_time
        p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
        median_kipp_time = float(np.median(kipp_times)) if len(kipp_times) > 0 else None

        # rent threshold and rent at kipp (if present)
        rent_threshold_vals = group['rent_threshold'].dropna().astype(
            float).tolist() if 'rent_threshold' in group else []
        rent_threshold = float(rent_threshold_vals[0]) if len(rent_threshold_vals) > 0 else None

        rent_at_kipp_vals = group['rent_at_kipp'].dropna().astype(float).tolist() if 'rent_at_kipp' in group else []
        rent_at_kipp = float(np.median(rent_at_kipp_vals)) if len(rent_at_kipp_vals) > 0 else None

        # investment / operational costs (take first non-null if available)
        invest_vals = group['investment_cost'].dropna().astype(float).tolist() if 'investment_cost' in group else []
        op_vals = group['annual_operational_cost'].dropna().astype(
            float).tolist() if 'annual_operational_cost' in group else []
        invest_cost_out = int(round(invest_vals[0])) if len(invest_vals) > 0 else np.nan
        op_cost_out = int(round(op_vals[0])) if len(op_vals) > 0 else np.nan

        # prepare outputs (rounded / nan where appropriate)
        p_kipp_out = round(p_kipp, 2)
        median_kipp_time_out = (round(median_kipp_time, 2) if median_kipp_time is not None else np.nan)
        rent_threshold_out = (round(rent_threshold, 2) if rent_threshold is not None else np.nan)
        rent_at_kipp_out = (round(rent_at_kipp, 2) if rent_at_kipp is not None else np.nan)

        results.append({
            'size': size,
            'quality': quality,
            'proximity': proximity,
            'function': park_function,
            'investment_cost': invest_cost_out,
            'annual_operational_cost': op_cost_out,
            'p_kipp': p_kipp_out,
            'median_kipp_time': median_kipp_time_out,
            'rent_threshold': rent_threshold_out,
            'rent_at_kipp': rent_at_kipp_out,
            'n_runs': n_runs_actual
        })

    # Save final aggregated results to CSV
    df_final = pd.DataFrame(results)
    final_path = os.path.join(out_dir, "grid_results.csv")
    tmp_final = os.path.join(out_dir, "grid_results.tmp.csv")
    df_final.to_csv(tmp_final, index=False)
    os.replace(tmp_final, final_path)

    # Optionally print a short summary
    print(f"Grid sweep finished: {len(df_runs)} runs aggregated into {len(df_final)} parameter combinations. -> {len(pd.DataFrame(raw_results))}")
    print(f"Final aggregated results written to: {final_path}")

    return df_final


# Visualization helpers
def plot_heatmap_from_grid(df_grid, x_col, y_col, value_col, out_file, x_log=False, y_log=False, cmap='viridis', title=None):
    """
    df_grid: DataFrame with columns x_col, y_col, value_col
    Produces a pivot heatmap and saves to out_file.
    """
    pivot = df_grid.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    pivot = pivot.reindex(index=sorted(df_grid[y_col].unique()), columns=sorted(df_grid[x_col].unique()))
    x_vals = sorted(df_grid[x_col].unique())
    y_vals = sorted(df_grid[y_col].unique())
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, origin='lower', aspect='auto', cmap=cmap)
    plt.colorbar(im, label=value_col)
    if x_log:
        xticklabels = [f"{v:.2e}" for v in x_vals]
    else:
        xticklabels = [f"{v:.0f}" for v in x_vals]
    plt.xticks(ticks=np.arange(len(x_vals)), labels=xticklabels, rotation=45)
    if y_log:
        yticklabels = [f"{v:.2e}" for v in y_vals]
    else:
        yticklabels = [f"{v:.2f}" for v in y_vals]
    plt.yticks(ticks=np.arange(len(y_vals)), labels=yticklabels)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title is not None else value_col)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_file) or ".")
    plt.savefig(out_file, dpi=150)
    plt.close()

# Example runner
def example_run():
    out_dir =  ExperimentConfig.out_dir
    ensure_dir(out_dir)

    run_start = time.perf_counter()

    size_values = [1000, 10000, 50000, 100000]
    quality_values = [0.2, 0.4, 0.6, 0.8]
    proximity_values = [0.5, 2.0, 4.0]
    function_values = ["recreation", "sports", "greenway"]

    config = ExperimentConfig(
        size_values=size_values,
        quality_values=quality_values,
        proximity_values=proximity_values,
        function_values=function_values,
    )

    df_grid = run_parameter_grid(config)

    write_run_statistics(df_grid)

    # If cost scenarios are active: only 2 heatmaps for the 4 cost variants
    if 'investment_cost' in df_grid.columns and 'annual_operational_cost' in df_grid.columns:
        df_costs = df_grid.dropna(subset=['investment_cost', 'annual_operational_cost']).copy()

        if not df_costs.empty:
            # Round costs to whole numbers (if not already rounded)
            df_costs['investment_cost'] = df_costs['investment_cost'].astype(float).round().astype(int)
            df_costs['annual_operational_cost'] = df_costs['annual_operational_cost'].astype(float).round().astype(int)

            # Aggregation: p_kipp mean, median_kipp_time median
            agg = df_costs.groupby(['investment_cost', 'annual_operational_cost']).agg(
                p_kipp_mean = ('p_kipp', 'mean'),
                median_kipp_time_med = ('median_kipp_time', lambda s: np.nanmedian(s.dropna().astype(float)))
            ).reset_index()

            # Pivot for Heatmaps: Investment (x) vs. Operations (y)
            pivot_p = agg.pivot_table(index='annual_operational_cost', columns='investment_cost', values='p_kipp_mean', aggfunc='mean')
            pivot_m = agg.pivot_table(index='annual_operational_cost', columns='investment_cost', values='median_kipp_time_med', aggfunc='mean')

            # Sort Axes (ascending)
            pivot_p = pivot_p.reindex(index=sorted(pivot_p.index), columns=sorted(pivot_p.columns))
            pivot_m = pivot_m.reindex(index=sorted(pivot_m.index), columns=sorted(pivot_m.columns))

            # Plot helper
            def save_cost_heatmap(pivot_df, title, cbar_label, fname, cmap='viridis'):
                plt.figure(figsize=(6, 5))
                im = plt.imshow(pivot_df.values, origin='lower', aspect='auto', cmap=cmap)
                cbar = plt.colorbar(im)
                cbar.set_label(cbar_label)
                x_vals = list(pivot_df.columns)
                y_vals = list(pivot_df.index)
                plt.xticks(ticks=np.arange(len(x_vals)), labels=[f"{int(x):,}" for x in x_vals], rotation=45)
                plt.yticks(ticks=np.arange(len(y_vals)), labels=[f"{int(y):,}" for y in y_vals])
                plt.xlabel("Investment cost (total) [$]")
                plt.ylabel("Annual operational cost (total) [$/Jahr]")
                plt.title(title)
                plt.tight_layout()
                ensure_dir(os.path.dirname(fname) or ".")
                plt.savefig(fname, dpi=150)
                plt.close()

            fname_p = os.path.join(f"{out_dir}/experiment_heatmaps", "heatmap_p_kipp_costs.png")
            fname_m = os.path.join(f"{out_dir}/experiment_heatmaps", "heatmap_median_kipp_time_costs.png")

            save_cost_heatmap(pivot_p, "p_kipp across cost scenarios", "p_kipp", fname_p, cmap='viridis')
            save_cost_heatmap(pivot_m, "median_kipp_time across cost scenarios", "median kipp time (years)", fname_m, cmap='magma')

            print(f"Cost Heatmaps Saved: {fname_p}, {fname_m}")
        else:
            # Standard heatmaps (when no cost scenarios are available)
            for prox in proximity_values:
                for func in function_values:
                    # use the correct column names
                    df_slice = df_grid[
                        (df_grid["proximity"] == prox) &
                        (df_grid["function"] == func)
                        ].copy()
                    if df_slice.empty:
                        print(f"[INFO] No data for proximity={prox}, function={func}.")
                        continue
                    # ensure numeric values
                    df_slice["median_kipp_time"] = pd.to_numeric(
                        df_slice["median_kipp_time"],
                        errors="coerce"
                    )
                    df_slice["p_kipp"] = pd.to_numeric(
                        df_slice["p_kipp"],
                        errors="coerce"
                    )
                    # -------------------------------------------------
                    # p_kipp heatmap
                    # -------------------------------------------------
                    if not df_slice["p_kipp"].dropna().empty:
                        out_file_p = os.path.join(
                            f"{out_dir}/experiment_heatmaps",
                            f"heatmap_p_kipp_prox_{prox}_func_{func}.png"
                        )
                        plot_heatmap_from_grid(
                            df_slice,
                            x_col="size",
                            y_col="quality",
                            value_col="p_kipp",
                            out_file=out_file_p,
                            x_log=False,
                            y_log=False,
                            cmap="viridis",
                            title=f"Probability of tipping (proximity = {prox}, function = {func})"
                        )
                        print(f"[HEATMAP] Saved: {out_file_p}")
                    # -------------------------------------------------
                    # median tipping time heatmap
                    # -------------------------------------------------
                    if df_slice["median_kipp_time"].dropna().empty:
                        print(f"[INFO] No median_kipp_time values for proximity={prox}, function={func}.")
                        continue
                    out_file_median = os.path.join(
                        f"{out_dir}/experiment_heatmaps",
                        f"heatmap_median_kipp_time_prox_{prox}_func_{func}.png"
                    )
                    plot_heatmap_from_grid(
                        df_slice,
                        x_col="size",
                        y_col="quality",
                        value_col="median_kipp_time",
                        out_file=out_file_median,
                        x_log=False,
                        y_log=False,
                        cmap="magma",
                        title=f"Median tipping time (proximity = {prox}, function = {func})"
                    )
                    print(f"[HEATMAP] Saved: {out_file_median}")
            print("Standard Heatmaps Created.")
    else:
        print("Cost fields not found in results; default heatmap logic remains active.")

    # Total runtime for example_run
    run_elapsed = time.perf_counter() - run_start
    hrs, rem = divmod(run_elapsed, 3600)
    mins, secs = divmod(rem, 60)
    run_elapsed_str = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
    print(f"Example run finished. Outputs in: {out_dir}. Total elapsed: {run_elapsed_str}")

if __name__ == "__main__":
    example_run()