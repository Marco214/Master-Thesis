"""
Global Sensitivity Analysis (GSA) for the Green Gentrification Model

Analyzes how strongly four park parameters influence two target outputs
of the model:

    - park_size      : park size in m²             (grid values: 1000/10000/50000/100000)
    - park_quality    : park quality, 0..1           (grid values: 0.2/0.4/0.6/0.8)
    - park_proximity : distance decay ("proximity")  (grid values: 0.5/2.0/4.0)
    - park_function  : park use (categorical)        (recreation/sports/greenway)

    1. Morris Screening (Elementary Effects, `SALib.sample/analyze.morris`)
       over all four parameters, target output `p_kipp` -> identifies
       the most influential parameters based on mu_star (mean absolute
       elementary effect).

    2. Sobol analysis (Saltelli sampling, `SALib.sample/analyze.sobol`)
       only for the parameters found most important by Morris (top-K),
       target output `p_kipp` -> quantifies main effects (S1) and
       interaction effects (S2, ST) on that tipping probability. The
       remaining, screening-unimportant parameters are held at a fixed
       reference value (drastically reduces the number of required
       simulations, since Sobol is very expensive: N*(2D+2) resp.
       N*(D+2) runs).
"""

import os
import time
import argparse
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol as sobol_analyzer

from experiments import run_single_experiment, ensure_dir, ExperimentConfig

# Parameter definition
PARK_FUNCTIONS = ["recreation", "sports", "greenway"]

# Names and bounds for the full 4-parameter GSA (Morris stage).
# Bounds correspond to the range of the original full-factorial values.
FULL_PROBLEM = {
    "num_vars": 4,
    "names": ["park_size", "park_quality", "park_proximity", "park_function"],
    "bounds": [
        [1000.0, 100000.0],                       # park_size [m²]
        [0.2, 0.8],                                # park_quality [0..1]
        [0.5, 4.0],                                 # park_proximity (= decay_scale)
        [0.0, len(PARK_FUNCTIONS) - 1e-9],          # park_function, encoded -> floor() -> index
    ],
}

# Reference values a parameter is fixed to when it is NOT among the
# top-K most influential parameters in the Sobol stage.
FIXED_REFERENCE = {
    "park_size": 10000.0,
    "park_quality": 0.5,
    "park_proximity": 2.0,
    "park_function": 0.0,     # -> "recreation"
}


def decode_function(code: float) -> str:
    """Map an encoded continuous value (0..3) to a park_function category."""
    idx = int(np.floor(code))
    idx = min(max(idx, 0), len(PARK_FUNCTIONS) - 1)
    return PARK_FUNCTIONS[idx]


# GSA configuration
@dataclass
class GSAConfig:
    n_replicates: int = 5                                  # repetitions per parameter sample (to average out stochastic noise)
    steps: int = ExperimentConfig.steps                     # simulation years per run
    rent_rel_threshold: float = ExperimentConfig.rent_rel_threshold
    income_shift_threshold: float = ExperimentConfig.income_shift_threshold
    persist_years: int = ExperimentConfig.persist_years
    model_kwargs: dict = field(default_factory=lambda: dict(ExperimentConfig.model_kwargs))
    base_seed: int = ExperimentConfig.base_seed
    n_workers: Optional[int] = None
    out_dir: str = "../output/gsa"


# Simulation worker (top-level, pickable for multiprocessing)
def _evaluate_row(task: Tuple[int, np.ndarray, GSAConfig, int]) -> Dict:
    """
    Runs several stochastic replicates of the model for ONE parameter
    combination (a row of the sample matrix X) and aggregates them into
    a single set of aggregated target values (p_kipp, median_kipp_time) per
    sample, as SALib expects one model output per sample.

    task: (sample_idx, row=[size, quality, proximity, func_code], cfg, seed_offset)
    """
    sample_idx, row, cfg, seed_offset = task
    park_size, park_quality, park_proximity, func_code = row
    park_function = decode_function(func_code)
    park_quality = float(np.clip(park_quality, 0.0, 1.0))

    kipp_flags = []
    kipp_times = []

    for rep in range(cfg.n_replicates):
        seed = cfg.base_seed + seed_offset + sample_idx * 1009 + rep

        params = {
            "park_size": float(park_size),
            "park_quality": park_quality,
            "decay_scale": float(park_proximity),
            "park_function": park_function,
            "model_base_kwargs": cfg.model_kwargs,
        }

        try:
            res = run_single_experiment(
                params,
                seed,
                cfg.steps,
                cfg.rent_rel_threshold,
                cfg.income_shift_threshold,
                cfg.persist_years,
                export=False,
            )
            kipp_flags.append(bool(res.get("kipp")))
            if res.get("kipp") and res.get("kipp_time") is not None:
                kipp_times.append(float(res["kipp_time"]))
        except Exception as e:
            # stay robust: one failed replicate should not derail the whole GSA;
            # it is counted as "no tipping"
            kipp_flags.append(False)

    p_kipp = (sum(kipp_flags) / len(kipp_flags)) if kipp_flags else 0.0

    if kipp_times:
        median_kipp_time = float(np.median(kipp_times))
    else:
        # right-censored: no tipping within the observation horizon
        median_kipp_time = float(cfg.steps)

    return {
        "sample_idx": sample_idx,
        "park_size": park_size,
        "park_quality": park_quality,
        "park_proximity": park_proximity,
        "park_function": park_function,
        "p_kipp": p_kipp,
        "median_kipp_time": median_kipp_time,
    }


def _run_batch(X: np.ndarray, cfg: GSAConfig, seed_offset: int, label: str) -> pd.DataFrame:
    """
    Evaluates all rows of X in parallel via multiprocessing and returns a
    DataFrame with one row per sample (order preserved, which is required
    by SALib.analyze).
    """
    n_samples = X.shape[0]
    tasks = [(i, X[i], cfg, seed_offset) for i in range(n_samples)]

    cpu_count = multiprocessing.cpu_count() or 1
    n_workers = cfg.n_workers or max(1, min(cpu_count, n_samples))
    n_workers = max(1, min(n_workers, cpu_count, n_samples))

    print(f"[{label}] {n_samples} parameter combinations  x  {cfg.n_replicates} replicates "
          f"= {n_samples * cfg.n_replicates} model runs  |  {n_workers} worker processes")

    results = [None] * n_samples
    start = time.perf_counter()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        chunksize = max(1, n_samples // (n_workers * 4))
        done = 0
        for res in pool.imap_unordered(_evaluate_row, tasks, chunksize=chunksize):
            results[res["sample_idx"]] = res
            done += 1
            if done % max(1, n_samples // 10) == 0 or done == n_samples:
                elapsed = time.perf_counter() - start
                print(f"[{label}] {done}/{n_samples} samples done ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - start
    print(f"[{label}] done in {elapsed:.1f}s")

    return pd.DataFrame(results)


def _expand_to_full(active_names: List[str], X_active: np.ndarray) -> np.ndarray:
    """
    Expands a sample matrix that only contains the 'active' (varied)
    parameters to the full 4D parameter space, by fixing the remaining
    parameters at their reference value (FIXED_REFERENCE).
    """
    n = X_active.shape[0]
    full_names = FULL_PROBLEM["names"]
    X_full = np.zeros((n, len(full_names)))
    for j, name in enumerate(full_names):
        if name in active_names:
            X_full[:, j] = X_active[:, active_names.index(name)]
        else:
            X_full[:, j] = FIXED_REFERENCE[name]
    return X_full

# Stage 1: Morris Screening
def run_morris_screening(cfg: GSAConfig, n_trajectories: int, num_levels: int,
                          seed: int, out_dir: str):
    print("\n" + "=" * 70)
    print("STAGE 1: MORRIS SCREENING (all 4 parameters)")
    print("=" * 70)

    problem = FULL_PROBLEM

    X = morris_sampler.sample(
        problem, N=n_trajectories, num_levels=num_levels, seed=seed
    )

    df = _run_batch(X, cfg, seed_offset=0, label="Morris")
    # Target output for the Morris screening is the tipping PROBABILITY
    # (p_kipp = share of stochastic replicates that reached a tipping
    # point), consistent with the Sobol stage further below.
    Y = df["p_kipp"].to_numpy(dtype=float)

    Si = morris_analyzer.analyze(
        problem, X, Y, num_levels=num_levels, print_to_console=True, seed=seed
    )

    # save results
    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, "morris_raw_results.csv"), index=False)

    df_indices = pd.DataFrame({
        "parameter": problem["names"],
        "mu": Si["mu"],
        "mu_star": Si["mu_star"],
        "mu_star_conf": Si["mu_star_conf"],
        "sigma": Si["sigma"],
    }).sort_values("mu_star", ascending=False).reset_index(drop=True)
    df_indices.to_csv(os.path.join(out_dir, "morris_indices.csv"), index=False)

    print("\nRanking by mu_star (influence on p_kipp, tipping probability):")
    print(df_indices.to_string(index=False))

    _plot_morris(df_indices, out_dir)

    return Si, df_indices, df


def _plot_morris(df_indices: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = df_indices.sort_values("mu_star")
    ax.barh(order["parameter"], order["mu_star"], xerr=order["mu_star_conf"],
            color="seagreen", ecolor="black", capsize=4)
    ax.set_xlabel(r"$\mu^*$ (mean absolute elementary effect on p_kipp)")
    ax.set_title("Morris Screening: Parameter Influence Ranking")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "morris_screening.png"), dpi=150)
    plt.close(fig)

    # mu* vs. sigma (nonlinearity / interactions)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(df_indices["mu_star"], df_indices["sigma"],
                xerr=df_indices["mu_star_conf"], fmt="o", color="darkorange",
                ecolor="gray", capsize=4)
    for _, row in df_indices.iterrows():
        ax.annotate(row["parameter"], (row["mu_star"], row["sigma"]),
                     textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel(r"$\mu^*$")
    ax.set_ylabel(r"$\sigma$ (spread -> nonlinearity/interactions)")
    ax.set_title("Morris: mu* vs. sigma")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "morris_mustar_vs_sigma.png"), dpi=150)
    plt.close(fig)

# Stage 2: Sobol analysis for the most important parameters
def run_sobol_analysis(cfg: GSAConfig, active_names: List[str], n_base_samples: int,
                        seed: int, out_dir: str, calc_second_order: bool = True):
    print("\n" + "=" * 70)
    print(f"STAGE 2: SOBOL ANALYSIS for top parameters: {active_names}")
    print("=" * 70)

    full_names = FULL_PROBLEM["names"]
    full_bounds = FULL_PROBLEM["bounds"]
    active_bounds = [full_bounds[full_names.index(n)] for n in active_names]

    reduced_problem = {
        "num_vars": len(active_names),
        "names": active_names,
        "bounds": active_bounds,
    }

    X_active = sobol_sampler.sample(
        reduced_problem, N=n_base_samples, calc_second_order=calc_second_order, seed=seed
    )
    X_full = _expand_to_full(active_names, X_active)

    df = _run_batch(X_full, cfg, seed_offset=500_000, label="Sobol")
    # Target output for the Sobol stage is the tipping PROBABILITY
    # (p_kipp = share of stochastic replicates that reached a tipping
    # point), not the median tipping time used in the Morris stage.
    Y = df["p_kipp"].to_numpy(dtype=float)

    Si = sobol_analyzer.analyze(
        reduced_problem, Y, calc_second_order=calc_second_order,
        print_to_console=True, seed=seed
    )

    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, "sobol_raw_results.csv"), index=False)

    df_s1_st = pd.DataFrame({
        "parameter": active_names,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)
    df_s1_st.to_csv(os.path.join(out_dir, "sobol_S1_ST.csv"), index=False)

    print("\nMain effects (S1) and total effects (ST) on p_kipp (tipping probability):")
    print(df_s1_st.to_string(index=False))

    df_s2 = None
    if calc_second_order and "S2" in Si:
        rows = []
        n = len(active_names)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "param_1": active_names[i],
                    "param_2": active_names[j],
                    "S2": Si["S2"][i][j],
                    "S2_conf": Si["S2_conf"][i][j],
                })
        df_s2 = pd.DataFrame(rows).sort_values("S2", ascending=False).reset_index(drop=True)
        df_s2.to_csv(os.path.join(out_dir, "sobol_S2_interactions.csv"), index=False)
        print("\nInteraction effects (S2) between parameter pairs (on p_kipp):")
        print(df_s2.to_string(index=False))

    _plot_sobol(df_s1_st, df_s2, out_dir)

    return Si, df_s1_st, df_s2, df


def _plot_sobol(df_s1_st: pd.DataFrame, df_s2: Optional[pd.DataFrame], out_dir: str):
    x = np.arange(len(df_s1_st))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, df_s1_st["S1"], width, yerr=df_s1_st["S1_conf"],
           label="S1 (main effect)", color="steelblue", capsize=4)
    ax.bar(x + width / 2, df_s1_st["ST"], width, yerr=df_s1_st["ST_conf"],
           label="ST (total effect)", color="indianred", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df_s1_st["parameter"])
    ax.set_ylabel("Sobol index")
    ax.set_title("Sobol: Main vs. Total Effects on p_kipp (Tipping Probability)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sobol_S1_ST.png"), dpi=150)
    plt.close(fig)

    if df_s2 is not None and not df_s2.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = df_s2["param_1"] + " x " + df_s2["param_2"]
        ax.bar(labels, df_s2["S2"], yerr=df_s2["S2_conf"], color="mediumpurple", capsize=4)
        ax.set_ylabel("S2 (interaction effect)")
        ax.set_title("Sobol: Interaction Effects Between Parameter Pairs")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sobol_S2_interactions.png"), dpi=150)
        plt.close(fig)

# Orchestration
def main():
    parser = argparse.ArgumentParser(
        description="Global Sensitivity Analysis (Morris -> Sobol) for the "
                     "Green Gentrification Model. Both stages target "
                     "p_kipp (tipping probability)."
    )
    parser.add_argument("--morris-n", type=int, default=20,
                         help="Number of Morris trajectories N (model runs: N*(D+1) samples). Default: 20")
    parser.add_argument("--morris-levels", type=int, default=4,
                         help="Number of levels in the Morris grid. Default: 4")
    parser.add_argument("--sobol-n", type=int, default=512,
                         help="Sobol base sample size N (model runs: N*(2D+2) samples). Default: 128")
    parser.add_argument("--top-k", type=int, default=2,
                         help="Number of parameters (ranked most important by Morris) that go into the "
                              "Sobol analysis; the rest are fixed. Default: 2")
    parser.add_argument("--replicates", type=int, default=5,
                         help="Stochastic repetitions per parameter combination. Default: 5")
    parser.add_argument("--steps", type=int, default=ExperimentConfig.steps,
                         help=f"Simulation years per run. Default: {ExperimentConfig.steps}")
    parser.add_argument("--n-agents", type=int, default=ExperimentConfig.model_kwargs["n_agents"],
                         help="Number of households in the model. Default matches ExperimentConfig.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of parallel worker processes. Default: all CPU cores.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the sampling procedures.")
    parser.add_argument("--out-dir", type=str, default="../output/gsa",
                         help="Output directory for CSVs and plots.")
    parser.add_argument("--no-second-order", action="store_true",
                         help="Do NOT compute Sobol S2 interaction indices (faster).")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    model_kwargs = dict(ExperimentConfig.model_kwargs)
    model_kwargs["n_agents"] = args.n_agents

    cfg = GSAConfig(
        n_replicates=args.replicates,
        steps=args.steps,
        model_kwargs=model_kwargs,
        n_workers=args.workers,
        out_dir=args.out_dir,
    )

    run_start = time.perf_counter()

    # --- Stage 1: Morris ---
    Si_morris, df_morris_indices, _ = run_morris_screening(
        cfg, n_trajectories=args.morris_n, num_levels=args.morris_levels,
        seed=args.seed, out_dir=args.out_dir,
    )

    # --- select the most important parameters for stage 2 ---
    top_k = max(1, min(args.top_k, len(FULL_PROBLEM["names"])))
    active_names = df_morris_indices["parameter"].head(top_k).tolist()
    print(f"\n-> Top-{top_k} parameters for Sobol analysis: {active_names}")
    print(f"   (remaining parameters fixed at: "
          f"{ {n: FIXED_REFERENCE[n] for n in FULL_PROBLEM['names'] if n not in active_names} })")

    # --- Stage 2: Sobol ---
    run_sobol_analysis(
        cfg, active_names=active_names, n_base_samples=args.sobol_n,
        seed=args.seed, out_dir=args.out_dir,
        calc_second_order=not args.no_second_order,
    )

    elapsed = time.perf_counter() - run_start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nGSA completed in {int(hrs):02d}:{int(mins):02d}:{secs:05.2f}. "
          f"Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
