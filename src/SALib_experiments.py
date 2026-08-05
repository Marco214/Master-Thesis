from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import numpy as np
from model import GreenGentModel
import random
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# -------------------------
# Problemdefinition
# -------------------------
problem = {
    'num_vars': 4,
    'names': ['size_values','quality_values','proximity_values','function_values'],
    'bounds': [[1000,100000],[0.2,0.8],[0.5,4.0],[1,3]]
}

# Basis-N (Sobol)
N = 1024
param_values = sample(problem, N)

# -------------------------
# Evaluate-Funktion (modell-spezifisch)
# -------------------------
def evaluate(X, steps=50, seed=None):
    """
    X: [size, quality, proximity, function_code]
    Liefert als Skalar die gewählte Zielgröße (hier: durchschnittliche Miete).
    """
    size = float(X[0])
    quality = float(X[1])
    proximity = float(X[2])
    func_idx = int(round(X[3]))
    func_idx = max(1, min(3, func_idx))
    function_map = {1: "recreation", 2: "sports", 3: "greenway"}
    function = function_map[func_idx]

    # Reproduzierbarkeit
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Modell instanziieren
    m = GreenGentModel(seed=seed)

    # Falls keine Parks vorhanden sind, gib NaN zurück
    if len(m.parks) == 0:
        return np.nan

    # Setze Park-Attribute entsprechend der Probe
    for p in m.parks:
        p.size = int(size)
        p.quality = float(quality)
        p.function = function
        if getattr(m, "enable_park_costs", False):
            p.compute_costs(m.cost_invest_per_m2,
                            m.cost_operational_per_m2_per_year,
                            m.quality_invest_multiplier,
                            m.quality_operational_multiplier)

    # Interpretiere 'proximity' als decay_scale für compute_green_scores
    m.compute_green_scores(decay_scale=float(proximity))

    # Modell laufen lassen (steps Jahre)
    m.run_model(steps)

    # Zielgröße: durchschnittliche Miete
    return m.average_rent()

# -------------------------
# Worker-Wrapper für Pool
# -------------------------
def _worker(args):
    idx, X, base_seed, steps = args
    seed = base_seed + idx
    try:
        y = evaluate(X, steps=steps, seed=seed)
    except Exception as e:
        # Fehlerbehandlung: protokolliere und gib NaN zurück
        print(f"[Worker {idx}] Fehler bei Sample {idx}: {e}")
        y = np.nan
    return idx, y

# -------------------------
# Plot-Funktion für Sobol Indices
# -------------------------
def plot_sobol_indices(Si, problem, figsize=(8,5), title="Sobol Sensitivity Indices", filename=None):
    names = problem['names']
    n = len(names)

    S1 = np.array([Si['S1'][i] for i in range(n)])
    ST = np.array([Si['ST'][i] for i in range(n)])
    S1_conf = np.array([Si.get('S1_conf', np.zeros_like(S1))[i] for i in range(n)])
    ST_conf = np.array([Si.get('ST_conf', np.zeros_like(ST))[i] for i in range(n)])

    ind = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(ind - width/2, S1, width, yerr=S1_conf, capsize=4, label='S1 (first order)', color='#4C72B0')
    bars2 = ax.bar(ind + width/2, ST, width, yerr=ST_conf, capsize=4, label='ST (total order)', color='#DD8452')

    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ymax = max(1.0, np.nanmax(np.concatenate([S1 + S1_conf, ST + ST_conf])) * 1.1)
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Sobol index')
    ax.set_title(title)
    ax.legend()

    for rect in list(bars1) + list(bars2):
        height = rect.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()

# -------------------------
# Main: Parallel ausführen, analysieren, plotten
# -------------------------
if __name__ == "__main__":
    base_seed = 42
    steps = 50

    tasks = [(i, param_values[i], base_seed, steps) for i in range(len(param_values))]

    n_procs = min(cpu_count(), len(tasks))
    print(f"Starte Pool mit {n_procs} Prozessen für {len(tasks)} Samples...")
    with Pool(processes=n_procs) as pool:
        results = pool.map(_worker, tasks)

    results.sort(key=lambda t: t[0])
    Y = np.array([t[1] for t in results])

    # Optional: entferne NaNs vor Analyse (oder behandle sie)
    valid_mask = ~np.isnan(Y)
    if not np.all(valid_mask):
        print(f"Achtung: {np.sum(~valid_mask)} fehlerhafte Läufe (NaN) — werden vor Analyse entfernt.")
    Y_valid = Y[valid_mask]
    param_values_valid = param_values[valid_mask]

    # SALib erwartet Y mit gleicher Länge wie Samples; wenn du Samples entfernst,
    # solltest du die Analyse mit der ursprünglichen sample-Matrix abstimmen.
    # Für schnelle Tests analysieren wir nur die gültigen Läufe (kleine N -> unsichere Indizes).
    Si = analyze(problem, Y_valid, print_to_console=True)
    print("First-order indices:", Si['S1'])
    print("Total-order indices:", Si['ST'])

    # Plot
    plot_sobol_indices(Si, problem, title="Sobol Indices (avg_rent)")
