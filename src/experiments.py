"""
Grid-sweep Experimente für Kipppunktanalyse der Gentrifizierung von urbanen Grünflächen
"""

import os
import time
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import GreenGentModel

# -------------------------
# Hilfsfunktionen
# -------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def safe_remove_from_list(lst, item):
    try:
        lst.remove(item)
    except ValueError:
        pass

def detect_tipping_local(model, baseline_rent,
                         park_pos=None,
                         neighborhood_radius=3,
                         rent_rel_threshold=1.10,
                         income_shift_threshold=0.003,
                         persist_years=2):
    """
    Kipppunktprüfung basierend auf lokalem Anteil High-Income-Haushalte
    in der Nachbarschaft eines Parks.

    Annahme: model.datacollector sammelt pro Timestep einen Reporter "high_local"
    (wie in deinem model.py: lambda m: m.local_high_income_share()).
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

    # -------------------------
    # Kipppunktprüfung
    # -------------------------
    for t in range(1, len(df)):
        rent_t = df['avg_rent'].iloc[t]

        # 1) Mietschwelle
        if rent_t >= baseline_rent_val * rent_rel_threshold:
            end_idx = min(len(df), t + int(persist_years))

            # 2) Persistenz
            if (df['avg_rent'].iloc[t:end_idx] >= baseline_rent_val * rent_rel_threshold).all():

                # 3) Lokaler sozialer Wandel
                high_change = share_high_local[t] - share_high_local[0] if len(share_high_local) > t else 0.0
                if high_change >= income_shift_threshold:
                    return True, t

    return False, None

# -------------------------
# Single-run wrapper
# -------------------------
def run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years, export=False):
    """
    params: dict with keys that will be passed to GreenGentModel (e.g., park_size, park_quality, decay_scale, park_function)
    seed: int
    returns: dict with run result (kipp, kipp_time, df)
    """
    # deterministische Seeds für Reproduzierbarkeit
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    model_base_kwargs = params.get('model_base_kwargs', {})
    model_kwargs = model_base_kwargs.copy()
    model_kwargs['seed'] = seed
    model = GreenGentModel(**model_kwargs)

    # Park-Override falls gewünscht
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
            ugs_cls = _UGS_fallback

        model.parks.append(ugs_cls((px, py), size, quality, function))

        decay = params.get('decay_scale', None)
        if hasattr(model, 'compute_green_scores'):
            model.compute_green_scores(decay_scale=decay)

        # sichere Entfernung aus alten Zellen
        for a in list(model.schedule.agents):
            old_pos = getattr(a, 'pos', None)
            if old_pos is None:
                continue
            occupants = model.cell_map.get(old_pos).occupants if old_pos in model.cell_map else []
            if a in occupants:
                safe_remove_from_list(occupants, a)

        # neue zufällige Positionen
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

    if df.empty or 'avg_rent' not in df.columns:
        return {'kipp': False, 'kipp_time': None, 'df': df}
    baseline_rent = df['avg_rent'].iloc[0]

    kipp, kipp_time = detect_tipping_local(
        model,
        baseline_rent=baseline_rent,
        park_pos=params.get('park_pos', model.parks[0].pos if model.parks else None),
        neighborhood_radius=params.get('neighborhood_radius', 3),
        rent_rel_threshold=rent_rel_threshold,
        income_shift_threshold=income_shift_threshold,
        persist_years=persist_years
    )

    if export:
        fname = os.path.join(params.get('out_dir', 'results'),
                             f"run_size{params.get('park_size')}_qual{params.get('park_quality')}_prox{params.get('decay_scale')}_func{params.get('park_function')}_seed{seed}.csv")
        ensure_dir(os.path.dirname(fname))
        df.to_csv(fname, index=False)

    return {'kipp': kipp, 'kipp_time': kipp_time, 'df': df}

# -------------------------
# Worker wrapper für multiprocessing (muss top-level sein)
# -------------------------
def _worker_task(task):
    """
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
        res = run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years, export=False)
        return {
            'park_size': size,
            'park_quality': quality,
            'decay_scale': proximity,
            'park_function': park_function,
            'run_idx': run_idx,
            'seed': seed,
            'kipp': res['kipp'],
            'kipp_time': res['kipp_time']
        }
    except Exception as e:
        # im Fehlerfall zurückgeben, damit Aggregation robust bleibt
        return {
            'park_size': size,
            'park_quality': quality,
            'decay_scale': proximity,
            'park_function': park_function,
            'run_idx': run_idx,
            'seed': seed,
            'kipp': False,
            'kipp_time': None,
            'error': str(e)
        }

# -------------------------
# Grid sweep orchestrator mit multiprocessing
# -------------------------
def run_parameter_grid(size_values, quality_values, proximity_values, function_values,
                       n_runs=1, steps=30,
                       rent_rel_threshold=1.10, income_shift_threshold=0.003, persist_years=2,
                       out_dir="results", model_base_kwargs=None, base_seed=42, n_workers=None):
    """
    Runs a 4D grid sweep over park size, quality, proximity (decay_scale) and park function.
    Parallelisiert die unabhängigen Monte-Carlo-Runs mit multiprocessing.
    """
    ensure_dir(out_dir)
    results = []

    # Erzeuge alle Tasks
    tasks = []
    job_counter = 0
    for size in size_values:
        for quality in quality_values:
            for proximity in proximity_values:
                for park_function in function_values:
                    for run_idx in range(n_runs):
                        seed = int(base_seed) + job_counter + run_idx
                        task = (size, quality, proximity, park_function, run_idx, seed,
                                model_base_kwargs or {}, steps, rent_rel_threshold, income_shift_threshold, persist_years, out_dir)
                        tasks.append(task)
                    job_counter += n_runs

    total_jobs = len(tasks)
    if total_jobs == 0:
        return pd.DataFrame(results)

    # Anzahl Worker bestimmen
    cpu_count = multiprocessing.cpu_count() or 1
    if n_workers is None:
        n_workers = max(1, min(cpu_count, total_jobs))
    else:
        n_workers = max(1, min(n_workers, cpu_count, total_jobs))

    # Verwende 'spawn' Kontext für bessere Kompatibilität (Windows)
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=n_workers)

    # chunksize heuristisch setzen
    chunksize = max(1, total_jobs // (n_workers * 4))

    raw_results = []
    save_every = max(1, total_jobs // 20)  # z.B. alle ~5% der Jobs speichern

    # Hilfsstruktur, um zu erkennen, welche Kombinationen bereits vollständig sind und bereits ausgegeben wurden
    completed_counts = {}  # key -> number of finished runs for that combo
    printed_combos = set()  # keys, die bereits ausgeprintet wurden

    try:
        for i, res in enumerate(pool.imap_unordered(_worker_task, tasks, chunksize)):
            raw_results.append(res)

            # Update finished-count für die jeweilige Parameter-Kombi
            key = (res['park_size'], res['park_quality'], res['decay_scale'], res['park_function'])
            completed_counts[key] = completed_counts.get(key, 0) + 1

            # Wenn alle erwarteten Runs für diese Kombination abgeschlossen sind und noch nicht ausgegeben wurden:
            if completed_counts[key] >= n_runs and key not in printed_combos:
                # Aggregiere Ergebnisse nur für diese Kombination aus raw_results
                combo_group = [r for r in raw_results
                               if (r['park_size'], r['park_quality'], r['decay_scale'], r['park_function']) == key]

                kipp_count = sum(1 for r in combo_group if r.get('kipp') == True)
                n_runs_actual = len(combo_group)
                kipp_times = [float(r['kipp_time']) for r in combo_group if
                              r.get('kipp') == True and r.get('kipp_time') is not None]
                p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                median_kipp_time = int(np.median(kipp_times)) if len(kipp_times) > 0 else None

                # Drucke eine kompakte Ergebniszeile
                print(f"[RESULT] size={key[0]:>7}  quality={key[1]:.2f}  decay={key[2]:.2f}  func={key[3]:>9}  "
                      f"p_kipp={p_kipp:.3f}  median_kipp_time={median_kipp_time}  runs={n_runs_actual}")

                # Markiere als ausgegeben
                printed_combos.add(key)

            # periodisch Zwischenergebnisse speichern (aggregiert pro Parameter-Kombination)
            if (i + 1) % save_every == 0 or (i + 1) == total_jobs:
                df_partial = pd.DataFrame(raw_results)

                # Aggregation auf Parameter-Kombinationsebene
                grouped = df_partial.groupby(['park_size', 'park_quality', 'decay_scale', 'park_function'])
                partial_rows = []
                for name, group in grouped:
                    size, quality, proximity, park_function = name
                    kipp_count = int(group['kipp'].sum())
                    n_runs_actual = len(group)
                    kipp_times = group.loc[group['kipp'] == True, 'kipp_time'].dropna().astype(float).tolist()
                    p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
                    median_kipp_time = int(np.median(kipp_times)) if len(kipp_times) > 0 else None
                    partial_rows.append({
                        'park_size': size,
                        'park_quality': quality,
                        'decay_scale': proximity,
                        'park_function': park_function,
                        'p_kipp': p_kipp,
                        'median_kipp_time': median_kipp_time,
                        'n_runs': n_runs_actual
                    })

                df_partial_out = pd.DataFrame(partial_rows)

                # atomar schreiben: zuerst temporär, dann ersetzen
                tmp_path = os.path.join(out_dir, "grid_results_partial.tmp.csv")
                df_partial_out.to_csv(tmp_path, index=False)
                os.replace(tmp_path, os.path.join(out_dir, "grid_results_partial.csv"))

    finally:
        pool.close()
        pool.join()

    # Aggregation: gruppiere nach Parameter-Kombination
    df_runs = pd.DataFrame(raw_results)
    # Falls Fehler-Spalte existiert, optional speichern
    if 'error' in df_runs.columns:
        err_df = df_runs[df_runs['error'].notnull()]
        if not err_df.empty:
            err_df.to_csv(os.path.join(out_dir, "grid_errors.csv"), index=False)

    grouped = df_runs.groupby(['park_size', 'park_quality', 'decay_scale', 'park_function'])

    for name, group in grouped:
        size, quality, proximity, park_function = name
        kipp_count = int(group['kipp'].sum())
        n_runs_actual = len(group)
        kipp_times = group.loc[group['kipp'] == True, 'kipp_time'].dropna().astype(float).tolist()
        p_kipp = kipp_count / n_runs_actual if n_runs_actual > 0 else 0.0
        median_kipp_time = int(np.median(kipp_times)) if len(kipp_times) > 0 else None
        results.append({
            'park_size': size,
            'park_quality': quality,
            'decay_scale': proximity,
            'park_function': park_function,
            'p_kipp': p_kipp,
            'median_kipp_time': median_kipp_time,
            'n_runs': n_runs_actual
        })
        # optional: save intermediate results
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(out_dir, "grid_results_partial.csv"), index=False)

    # final save
    df_final = pd.DataFrame(results)
    df_final.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)
    elapsed = time.time() - (min([t[5] for t in tasks]) if tasks else time.time())  # dummy baseline
    print(f"Grid sweep finished. Total jobs: {total_jobs}. Results saved to {out_dir}")
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
    plt.title(f"{value_col} over {x_col} x {y_col}")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_file) or ".")
    plt.savefig(out_file, dpi=150)
    plt.close()

# -------------------------
# Example runner (CLI style)
# -------------------------
def example_run():
    out_dir = "../output/experiments"
    ensure_dir(out_dir)

    size_values = [1000, 5000, 10000, 50000, 100000]
    quality_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    proximity_values = [0.5, 1.0, 2.0, 4.0]  # decay_scale
    function_values = ["recreation", "sports", "greenway"]

    df_grid = run_parameter_grid(size_values, quality_values, proximity_values, function_values,
                                 n_runs=1, steps=30,
                                 rent_rel_threshold=1.10, income_shift_threshold=0.003, persist_years=2,
                                 out_dir=out_dir, model_base_kwargs={'width':7, 'height':7, 'n_agents':1000}, base_seed=42, n_workers=None)

    # plot 2D slices: p_kipp and median_kipp_time over size x quality for each proximity and function
    for prox in proximity_values:
        for func in function_values:
            df_slice = df_grid[(df_grid['decay_scale'] == prox) & (df_grid['park_function'] == func)]
            if df_slice.empty:
                continue

            # gemeinsame Vorverarbeitung
            df_slice = df_slice.copy()
            # p_kipp sollte numerisch vorliegen; median_kipp_time: None -> NaN
            df_slice['median_kipp_time'] = df_slice['median_kipp_time'].apply(lambda x: np.nan if x is None else x)

            # 1) Heatmap p_kipp
            out_file_p = os.path.join(out_dir, f"heatmap_p_kipp_prox_{prox}_func_{func}.png")
            plot_heatmap_from_grid(df_slice, x_col='park_size', y_col='park_quality', value_col='p_kipp',
                                   out_file=out_file_p, x_log=False, y_log=False, cmap='viridis')
            print(f"[HEATMAP] p_kipp heatmap saved: {out_file_p}")

            # 2) Heatmap median_kipp_time
            # Prüfen, ob es überhaupt median-Werte gibt
            if df_slice['median_kipp_time'].dropna().empty:
                print(f"[INFO] Keine median_kipp_time Werte für prox={prox}, func={func}; übersprungen.")
                continue

            # Optional: automatische vmin/vmax anhand Quantile für bessere Kontraste
            vmin = float(df_slice['median_kipp_time'].dropna().quantile(0.05))
            vmax = float(df_slice['median_kipp_time'].dropna().quantile(0.95))

            out_file_median = os.path.join(out_dir, f"heatmap_median_kipp_time_prox_{prox}_func_{func}.png")
            # plot_heatmap_from_grid akzeptiert derzeit kein vmin/vmax-Argument;
            # falls du vmin/vmax nutzen willst, erweitere plot_heatmap_from_grid oder setze sie global.
            plot_heatmap_from_grid(df_slice, x_col='park_size', y_col='park_quality',
                                   value_col='median_kipp_time', out_file=out_file_median,
                                   x_log=False, y_log=False, cmap='magma')

    print("Example run finished. Outputs in:", out_dir)

if __name__ == "__main__":
    example_run()
