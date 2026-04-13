import os
import csv
import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import GreenGentModel  # dein Modell
# from multiprocessing import Pool  # optional für Parallelisierung

# -------------------------
# Hilfsfunktionen
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def detect_tipping_from_df(df, baseline_rent, rent_rel_threshold=1.25,
                           income_shift_threshold=0.15, persist_years=3):
    """
    Detect tipping event in model time series dataframe.
    df: DataFrame mit Spalten 'avg_rent', 'low_count', 'middle_count', 'high_count'
    baseline_rent: Referenzwert (z.B. avg_rent zu t0)
    Returns: (kipp_bool, kipp_time_step)
    """
    # compute share_high time series
    total = df['low_count'] + df['middle_count'] + df['high_count']
    # avoid division by zero
    share_high = (df['high_count'] / total).fillna(0.0)

    for t in range(1, len(df)):
        rent_t = df['avg_rent'].iloc[t]
        if rent_t >= baseline_rent * rent_rel_threshold:
            # check persistence
            end_idx = min(len(df), t + persist_years)
            if (df['avg_rent'].iloc[t:end_idx] >= baseline_rent * rent_rel_threshold).all():
                # check social shift
                if (share_high.iloc[t] - share_high.iloc[0]) >= income_shift_threshold:
                    return True, t
    return False, None

# -------------------------
# Single-run wrapper
# -------------------------
def run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years, export=False):
    """
    params: dict with keys that will be passed to GreenGentModel (e.g., park_size, park_quality, decay_scale)
    seed: int
    returns: dict with run result (kipp, kipp_time, time_series_df)
    """
    # map param names to model constructor args
    model_kwargs = {}
    # expected param names: park_size, park_quality, decay_scale (decay_scale used in compute_green_scores)
    # We set parks manually after model init if needed.
    model = GreenGentModel(seed=seed, **params.get('model_base_kwargs', {}))

    # If user wants to override park attributes, do it here:
    # Example: replace parks with single park of given size/quality at center
    if 'park_size' in params or 'park_quality' in params or 'park_pos' in params or 'decay_scale' in params:
        # simple strategy: clear parks and create one park centroid with given attributes
        model.parks = []
        px, py = params.get('park_pos', (model.width // 2, model.height // 2))
        size = params.get('park_size', 10000)
        quality = params.get('park_quality', 0.7)
        function = params.get('park_function', random.choice(["recreation", "sports", "greenway"]))
        from model import UGS as UGS_local  # if UGS class is in separate file; else import from model
        try:
            # try to use model's UGS class if available
            UGS_cls = getattr(model, 'UGS', None)
            if UGS_cls is None:
                UGS_cls = UGS_local
        except Exception:
            UGS_cls = UGS_local
        model.parks.append(UGS_cls((px, py), size, quality, function))
        # recompute green scores with optional decay_scale
        decay = params.get('decay_scale', None)
        model.compute_green_scores(decay_scale=decay)

    # run model
    model.run_model(steps)

    # collect time series
    df = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    # baseline rent at t0
    if df.empty:
        return {'kipp': False, 'kipp_time': None, 'df': df}
    baseline_rent = df['avg_rent'].iloc[0]

    kipp, kipp_time = detect_tipping_from_df(df,
                                             baseline_rent=baseline_rent,
                                             rent_rel_threshold=rent_rel_threshold,
                                             income_shift_threshold=income_shift_threshold,
                                             persist_years=persist_years)
    if export:
        # attach params to df and save
        fname = os.path.join(params.get('out_dir', 'results'), f"run_seed_{seed}.csv")
        ensure_dir(os.path.dirname(fname))
        df.to_csv(fname, index=False)

    return {'kipp': kipp, 'kipp_time': kipp_time, 'df': df}

# -------------------------
# Grid sweep orchestrator
# -------------------------
def run_parameter_grid(size_values, quality_values, proximity_values,
                       n_runs=1, steps=10,
                       rent_rel_threshold=1.25, income_shift_threshold=0.15, persist_years=3,
                       out_dir="results", model_base_kwargs=None):
    """
    Runs a 3D grid sweep over park size, quality and proximity (decay_scale).
    For each grid cell, runs n_runs Monte Carlo simulations with different seeds.
    Saves aggregated results to CSV and produces heatmaps for 2D slices.
    """
    ensure_dir(out_dir)
    results = []
    total_jobs = len(size_values) * len(quality_values) * len(proximity_values) * n_runs
    job_counter = 0
    start_time = time.time()

    for size in size_values:
        for quality in quality_values:
            for proximity in proximity_values:
                kipp_count = 0
                kipp_times = []
                for run_idx in range(n_runs):
                    seed = int(time.time() * 1000) % 2**31 + run_idx + job_counter
                    params = {
                        'park_size': size,
                        'park_quality': quality,
                        'decay_scale': proximity,
                        'model_base_kwargs': model_base_kwargs or {},
                        'out_dir': out_dir
                    }
                    res = run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years, export=False)
                    if res['kipp']:
                        kipp_count += 1
                        kipp_times.append(res['kipp_time'])
                    job_counter += 1

                p_kipp = kipp_count / n_runs
                median_kipp_time = np.median(kipp_times) if len(kipp_times) > 0 else None
                results.append({
                    'park_size': size,
                    'park_quality': quality,
                    'decay_scale': proximity,
                    'p_kipp': p_kipp,
                    'median_kipp_time': median_kipp_time,
                    'n_runs': n_runs
                })
                # optional: save intermediate results
                df_res = pd.DataFrame(results)
                df_res.to_csv(os.path.join(out_dir, "grid_results_partial.csv"), index=False)

    # final save
    df_final = pd.DataFrame(results)
    df_final.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)
    elapsed = time.time() - start_time
    print(f"Grid sweep finished in {elapsed:.1f}s. Results saved to {out_dir}")
    return df_final

# -------------------------
# Visualization helpers
# -------------------------
def plot_heatmap_from_grid(df_grid, x_col, y_col, value_col, out_file, x_log=False, y_log=False, cmap='viridis'):
    """
    df_grid: DataFrame with columns x_col, y_col, value_col
    Produces a pivot heatmap and saves to out_file.
    """
    pivot = df_grid.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    x_vals = sorted(df_grid[x_col].unique())
    y_vals = sorted(df_grid[y_col].unique())
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, origin='lower', aspect='auto', cmap=cmap)
    plt.colorbar(im, label=value_col)
    plt.xticks(ticks=np.arange(len(x_vals)), labels=[f"{v:.0f}" for v in x_vals], rotation=45)
    plt.yticks(ticks=np.arange(len(y_vals)), labels=[f"{v:.2f}" for v in y_vals])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{value_col} over {x_col} x {y_col}")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

# -------------------------
# Example runner (CLI style)
# -------------------------
def example_run():
    out_dir = "../output/experiments"
    ensure_dir(out_dir)

    # coarse grid (angepasst an dein Modell)
    size_values = [1000, 5000, 10000, 50000, 100000]
    quality_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    proximity_values = [0.5, 1.0, 2.0, 4.0]  # decay_scale

    df_grid = run_parameter_grid(size_values, quality_values, proximity_values,
                                 n_runs=1, steps=10,
                                 rent_rel_threshold=1.25, income_shift_threshold=0.15, persist_years=3,
                                 out_dir=out_dir, model_base_kwargs={'width':7, 'height':7, 'n_agents':1000})

    # plot a 2D slice: p_kipp over size x quality for a fixed proximity
    prox = proximity_values[1]
    df_slice = df_grid[df_grid['decay_scale'] == prox]
    plot_heatmap_from_grid(df_slice, x_col='park_size', y_col='park_quality', value_col='p_kipp',
                           out_file=os.path.join(out_dir, f"heatmap_p_kipp_prox_{prox}.png"))

    print("Example run finished. Outputs in:", out_dir)

# -------------------------
# If executed as script
# -------------------------
if __name__ == "__main__":
    example_run()
