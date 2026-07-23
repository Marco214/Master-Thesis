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
                         persist_years=3):
    """
    Kipppunktprüfung basierend auf lokalem Anteil High-Income-Haushalte
    in der Nachbarschaft eines Parks.
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
    returns: dict with run result (kipp, kipp_time, df, investment_cost, annual_operational_cost)
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
                    self.investment_cost = 0.0
                    self.annual_operational_cost = 0.0
            ugs_cls = _UGS_fallback

        # append park
        new_park = ugs_cls((px, py), size, quality, function)
        model.parks.append(new_park)

        # --- WICHTIG: Falls das Modell Kosten aktiviert hat, berechne Kosten für den neuen Park
        if getattr(model, "enable_park_costs", False):
            invest_per_m2 = getattr(model, "cost_invest_per_m2", None)
            op_per_m2 = getattr(model, "cost_operational_per_m2_per_year", None)
            q_inv_mult = getattr(model, "quality_invest_multiplier", 0.0)
            q_op_mult = getattr(model, "quality_operational_multiplier", 0.0)

            if hasattr(new_park, "compute_costs") and invest_per_m2 is not None and op_per_m2 is not None:
                new_park.compute_costs(invest_per_m2, op_per_m2, q_inv_mult, q_op_mult)
            else:
                # Fallback: set Attribute manuell, falls compute_costs nicht vorhanden
                new_park.investment_cost = new_park.size * (invest_per_m2 if invest_per_m2 is not None else 0.0) * (
                            1.0 + new_park.quality * q_inv_mult)
                new_park.annual_operational_cost = new_park.size * (op_per_m2 if op_per_m2 is not None else 0.0) * (
                            1.0 + new_park.quality * q_op_mult)

            # Aktualisiere aggregierte Summen im Model
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

    # --- Gesamtkosten aus dem Modell lesen (falls vorhanden) ---
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
        neighborhood_radius=params.get('neighborhood_radius', 3),
        rent_rel_threshold=rent_rel_threshold,
        income_shift_threshold=income_shift_threshold,
        persist_years=persist_years
    )

    # berechne Mietschwelle und Mietwert zum Kippzeitpunkt (falls vorhanden)
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
        res = run_single_experiment(params, seed, steps, rent_rel_threshold, income_shift_threshold, persist_years,
                                    export=False)

        # Lese die vom Modell berechneten Summen (können None sein)
        total_inv = res.get('investment_cost', None)
        total_op = res.get('annual_operational_cost', None)

        # Runde auf ganze Zahlen, falls Werte vorhanden
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
        # Falls ein Fehler auftritt, gib eine robuste Struktur zurück (ohne Zugriff auf res)
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


# -------------------------
# Grid sweep orchestrator mit multiprocessing
# -------------------------
def run_parameter_grid(size_values, quality_values, proximity_values, function_values,
                       n_runs=1, steps=50,
                       rent_rel_threshold=1.10, income_shift_threshold=0.003, persist_years=3,
                       out_dir="results", model_base_kwargs=None, base_seed=42, n_workers=None,
                       use_multiprocessing=False):
    """
    Runs a 4D grid sweep. Set use_multiprocessing=False to run sequentially.
    """
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

    # --- Sequenzieller Modus ---
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
                # gleiche Aggregations- und Speicherroutine wie im Original
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

    # --- Multiprocessing Modus (originale Logik) ---
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

    # --- gemeinsame Abschlussaggregation wie zuvor ---
    df_runs = pd.DataFrame(raw_results)
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
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(out_dir, "grid_results_partial.csv"), index=False)

    df_final = pd.DataFrame(results)
    df_final.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)
    elapsed = time.perf_counter() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
    print(f"Grid sweep finished. Total jobs: {total_jobs}. Elapsed time: {elapsed_str}. Results saved to {out_dir}")

    try:
        info_path = os.path.join(out_dir, "run_info.txt")
        with open(info_path, "w") as fh:
            fh.write(f"finished_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"total_jobs: {total_jobs}\n")
            fh.write(f"elapsed_seconds: {elapsed:.4f}\n")
            fh.write(f"elapsed_hms: {elapsed_str}\n")
    except Exception:
        pass

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

    run_start = time.perf_counter()

    size_values = [1000, 10000, 50000, 100000]
    quality_values = [0.2, 0.4, 0.6, 0.8]
    proximity_values = [0.5, 2.0, 4.0]
    function_values = ["recreation", "sports", "greenway"]

    df_grid = run_parameter_grid(size_values, quality_values, proximity_values, function_values,
                                 n_runs=1, steps=50,
                                 rent_rel_threshold=1.10, income_shift_threshold=0.003, persist_years=3,
                                 out_dir=out_dir, model_base_kwargs={'width':7, 'height':7, 'n_agents':1000, 'enable_park_costs': False}, base_seed=42, n_workers=None)

    # -------------------------
    # Wenn Kosten-Szenarien aktiv sind: nur 2 Heatmaps für die 4 Kostenvarianten
    # -------------------------
    if 'investment_cost' in df_grid.columns and 'annual_operational_cost' in df_grid.columns:
        df_costs = df_grid.dropna(subset=['investment_cost', 'annual_operational_cost']).copy()

        if not df_costs.empty:
            # Runde Kosten auf ganze Zahlen (falls nicht bereits gerundet)
            df_costs['investment_cost'] = df_costs['investment_cost'].astype(float).round().astype(int)
            df_costs['annual_operational_cost'] = df_costs['annual_operational_cost'].astype(float).round().astype(int)

            # Aggregation: p_kipp mean, median_kipp_time median
            agg = df_costs.groupby(['investment_cost', 'annual_operational_cost']).agg(
                p_kipp_mean = ('p_kipp', 'mean'),
                median_kipp_time_med = ('median_kipp_time', lambda s: np.nanmedian(s.dropna().astype(float)))
            ).reset_index()

            # Pivot für Heatmaps: Investition (x) vs Betrieb (y)
            pivot_p = agg.pivot_table(index='annual_operational_cost', columns='investment_cost', values='p_kipp_mean', aggfunc='mean')
            pivot_m = agg.pivot_table(index='annual_operational_cost', columns='investment_cost', values='median_kipp_time_med', aggfunc='mean')

            # Sortiere Achsen (aufsteigend)
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

            # Dateinamen
            fname_p = os.path.join(out_dir, "heatmap_p_kipp_costs.png")
            fname_m = os.path.join(out_dir, "heatmap_median_kipp_time_costs.png")

            # Speichern (p_kipp: 0..1, median_kipp_time: Jahre)
            save_cost_heatmap(pivot_p, "Kipp-Wahrscheinlichkeit über Kostenvarianten", "p_kipp (Wahrscheinlichkeit)", fname_p, cmap='viridis')
            save_cost_heatmap(pivot_m, "Median Kipp-Zeit über Kostenvarianten", "Median Kipp-Zeit (Jahre)", fname_m, cmap='magma')

            print(f"[INFO] Kosten-Heatmaps gespeichert: {fname_p}, {fname_m}")
        else:
            print("[WARN] Keine Runs mit investment_cost/annual_operational_cost gefunden; keine Kosten-Heatmaps erstellt.")
    else:
        # Fallback: falls keine Kosten-Spalten vorhanden sind, behalte das alte Verhalten
        print("[INFO] Kostenfelder nicht in Ergebnissen gefunden; Standard-Heatmap-Logik bleibt aktiv.")
        # Optional: hier könntest du die alte Schleife wieder aktivieren, falls gewünscht.

    # Gesamtlaufzeit für example_run
    run_elapsed = time.perf_counter() - run_start
    hrs, rem = divmod(run_elapsed, 3600)
    mins, secs = divmod(rem, 60)
    run_elapsed_str = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
    print(f"Example run finished. Outputs in: {out_dir}. Total elapsed: {run_elapsed_str}")

if __name__ == "__main__":
    example_run()
