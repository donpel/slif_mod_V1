"""
Experimental plywood sanding analysis and simple regression-based simulation.

The script builds a pandas DataFrame from provided belt sanding experiments,
fits a linear model for material removal rate (MRR) versus normal force,
plots relationships, and simulates removal for new force scenarios.
"""
from __future__ import annotations

import math
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend so the script can run headless and save PNGs.
matplotlib.use("Agg")


def create_experiment_dataframe() -> pd.DataFrame:
    """Create DataFrame with experimental data and derived columns."""

    data = [
        {"weight_g": 451, "time_s": 54, "Ft_N": 1.578082192, "FN_N": 4.42431, "speed_mm_min": 5.555555556},
        {"weight_g": 1100, "time_s": 28, "Ft_N": 6.575342466, "FN_N": 10.791, "speed_mm_min": 10.71428571},
        {"weight_g": 1400, "time_s": 23, "Ft_N": 8.153424658, "FN_N": 13.734, "speed_mm_min": 13.04347826},
        {"weight_g": 1915, "time_s": 16, "Ft_N": 9.468493151, "FN_N": 18.78615, "speed_mm_min": 18.75},
        {"weight_g": 2212, "time_s": 13, "Ft_N": 14.72876712, "FN_N": 21.69972, "speed_mm_min": 23.07692308},
        {"weight_g": 5000, "time_s": 6, "Ft_N": 29.19452055, "FN_N": 49.05, "speed_mm_min": 50.0},
    ]

    df = pd.DataFrame(data)

    # Derived metrics.
    df["MRR_mm_per_min"] = 5.0 / df["time_s"] * 60.0
    df["MRR_mm_per_s"] = df["MRR_mm_per_min"] / 60.0
    df["ratio_Ft_to_FN"] = df["Ft_N"] / df["FN_N"]

    return df


def fit_mrr_vs_FN(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Fit linear model MRR = a * FN + b and return coefficients and R^2."""

    x = df["FN_N"].to_numpy()
    y = df["MRR_mm_per_min"].to_numpy()
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(a), float(b), float(r2)


def plot_mrr_vs_FN(df: pd.DataFrame, a: float, b: float, filename: str = "mrr_vs_FN.png") -> None:
    """Scatter plot of MRR vs FN with fitted line."""

    plt.figure(figsize=(7, 5))
    plt.scatter(df["FN_N"], df["MRR_mm_per_min"], color="tab:blue", label="Experiment")

    x_line = np.linspace(df["FN_N"].min() * 0.9, df["FN_N"].max() * 1.1, 200)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color="tab:red", label=f"Fit: MRR = {a:.3f} * FN + {b:.3f}")

    plt.xlabel("Normal force FN [N]")
    plt.ylabel("MRR [mm/min]")
    plt.title("Material removal rate vs normal force")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def plot_Ft_vs_FN(df: pd.DataFrame, filename: str = "Ft_vs_FN.png") -> None:
    """Scatter plot of cutting force vs normal force."""

    plt.figure(figsize=(7, 5))
    plt.scatter(df["FN_N"], df["Ft_N"], color="tab:green", label="Experiment")
    plt.xlabel("Normal force FN [N]")
    plt.ylabel("Tangential force Ft [N]")
    plt.title("Cutting force vs normal force")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def plot_ratio_vs_FN(df: pd.DataFrame, filename: str = "ratio_vs_FN.png") -> None:
    """Scatter plot of force ratio vs normal force with mean line."""

    plt.figure(figsize=(7, 5))
    plt.scatter(df["FN_N"], df["ratio_Ft_to_FN"], color="tab:purple", label="Ft/FN")
    mean_ratio = df["ratio_Ft_to_FN"].mean()
    plt.axhline(mean_ratio, color="tab:orange", linestyle="--", label=f"Mean ratio = {mean_ratio:.3f}")
    plt.xlabel("Normal force FN [N]")
    plt.ylabel("Ft / FN [-]")
    plt.title("Force ratio vs normal force")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def simulate_scenario(a: float, b: float) -> None:
    """Simulate sanding removal for specified normal forces using fitted model."""

    h_edge_mm = 11.0
    L_contact_mm = 100.0
    h_edge_cm = h_edge_mm / 10.0
    L_contact_cm = L_contact_mm / 10.0
    A_contact_cm2 = h_edge_cm * L_contact_cm

    t_sim = 30.0  # seconds
    FN_cases = [18.78615, 21.69972]

    print("\nSimulation results (30 s) using fitted MRR = a * FN + b:")
    for FN_sim in FN_cases:
        mrr_sim_mm_min = a * FN_sim + b
        mrr_sim_mm_s = mrr_sim_mm_min / 60.0
        z_sim_mm = mrr_sim_mm_s * t_sim
        z_sim_cm = z_sim_mm / 10.0
        V_sim_cm3 = A_contact_cm2 * z_sim_cm
        m_sim_g = V_sim_cm3 * 0.68  # density

        print(f"\nFN = {FN_sim:.5f} N")
        print(f"  MRR_sim = {mrr_sim_mm_min:.3f} mm/min ({mrr_sim_mm_s:.4f} mm/s)")
        print(f"  Removed thickness after {t_sim:.0f} s: {z_sim_mm:.3f} mm")
        print(f"  Removed volume: {V_sim_cm3:.3f} cm^3")
        print(f"  Removed mass (rho=0.68 g/cm^3): {m_sim_g:.3f} g")

        # Optional comparison to experimental value if available.
        matching_row = None
        if math.isclose(FN_sim, 18.78615, rel_tol=1e-6):
            matching_row = 18.78615
        elif math.isclose(FN_sim, 21.69972, rel_tol=1e-6):
            matching_row = 21.69972

        if matching_row is not None:
            exp_table = create_experiment_dataframe()
            exp_row = exp_table.loc[np.isclose(exp_table["FN_N"], matching_row)].iloc[0]
            mrr_exp = float(exp_row["MRR_mm_per_min"])
            abs_err = mrr_sim_mm_min - mrr_exp
            rel_err = abs_err / mrr_exp if mrr_exp != 0 else float("nan")
            print(f"  Experimental MRR: {mrr_exp:.3f} mm/min")
            print(f"  Error: {abs_err:.3f} mm/min ({rel_err:.1%})")


def main():
    df = create_experiment_dataframe()
    print("Experimental data with derived columns:")
    print(df.to_string(index=False))

    a, b, r2 = fit_mrr_vs_FN(df)
    print(f"\nFitted MRR = a * FN + b: a = {a:.4f}, b = {b:.4f}, R^2 = {r2:.4f}")

    plot_mrr_vs_FN(df, a, b)
    plot_Ft_vs_FN(df)
    plot_ratio_vs_FN(df)
    print("Saved plots: mrr_vs_FN.png, Ft_vs_FN.png, ratio_vs_FN.png")

    simulate_scenario(a, b)


if __name__ == "__main__":
    main()
