"""
# Example run and plotting
"""
import glob
import os
import numpy as np
from matplotlib import pyplot as plt

from model import GreenGentModel


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def clear_out_dir(out_dir, pattern="*.png"):
    """Safely remove only matching files (default PNGs) from out_dir."""
    if not os.path.exists(out_dir):
        return
    files = glob.glob(os.path.join(out_dir, pattern))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warnung: Datei {f} konnte nicht gelöscht werden: {e}")

# Example run and plots
def run_example(steps, seed):
    out_dir = "../output/heatmaps"
    ensure_dir(out_dir)
    clear_out_dir(out_dir, pattern="*.png")  # clear previous PNGs

    model = GreenGentModel(width=7, height=7, n_agents=1000,
                          seed=seed, export_every=10, out_dir=out_dir,
                          w_proximity=0.40, w_size=0.25, w_quality=0.25, w_function=0.10,
                          beta_ugs=0.12, park_coverage=0.12)

    # export initial maps
    model.export_heatmaps(0)

    # save a standalone green map for reference
    green_grid = model.build_green_grid()
    model.save_heatmap(green_grid, "Green Score (initial)", f"Green Score", os.path.join(model.out_dir, "green_map_initial.png"),
                       cmap="Greens", vmin=0.0, vmax=1.0, overlay_parks=True)

    avg_incomes, avg_rents, avg_green = [], [], []
    for i in range(steps):
        model.step()
        if (i + 1) % model.export_every == 0:
            model.export_heatmaps(i + 1)

        # Data from DataCollector (avg_rent, avg_green) if available
        df = model.datacollector.get_model_vars_dataframe()

        if not df.empty:
            rec = df.iloc[-1]
            avg_rents.append(rec["avg_rent"])
            avg_green.append(rec["avg_green"])
        else:
            avg_rents.append(np.nan)
            avg_green.append(np.nan)

        # calculate average income
        incomes = [a.income_value for a in model.schedule.agents]
        if len(incomes) > 0:
            avg_incomes.append(float(np.mean(incomes)))
        else:
            avg_incomes.append(np.nan)

    # Plot: Average income over time
    """plt.figure(figsize=(8, 4))
    plt.plot(avg_incomes, label="avg_income", color="tab:blue")
    plt.title("Average income over time")
    plt.xlabel("Years")
    plt.ylabel("Average income per month")
    plt.legend()
    plt.tight_layout()"""

    # Plot: Average rent over time
    plt.figure(figsize=(8, 4))
    plt.plot(avg_rents, label="avg_rent", color="tab:orange")
    plt.title("Average rent over time")
    plt.xlabel("Years")
    plt.ylabel("Rent per month")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    run_example(steps=50, seed=122)