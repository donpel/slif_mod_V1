"""
Dynamic model, simulation, and benchmarking toolkit for plywood edge sanding.
Implements a four-state ODE with helper utilities for scenario sweeps and plotting.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Use non-interactive backend for safer script execution.
matplotlib.use("Agg")


@dataclass
class ModelParameters:
    """Model parameter container with modest default values."""

    k_m: float = 0.08
    a_p: float = 1.0
    a_v: float = 0.4

    k_T: float = 25.0
    T_env: float = 20.0
    tau_T: float = 1.6

    k_w: float = 0.6

    k_R: float = 4.5
    gamma_W: float = 0.6
    tau_R: float = 0.8

    grit_ref: float = 120.0
    n_grit_m: float = 0.35
    n_grit_T: float = 0.1
    n_grit_R: float = 0.9

    n_hard_T: float = 0.35
    n_hard_R: float = 0.25


@dataclass
class Inputs:
    """Input operating conditions for a simulation run."""

    theta_deg: float
    pressure: float
    v_rel: float
    hardness: float
    grit: float


def f_theta_m(theta_deg: float) -> float:
    """Angle influence on material interactions."""

    rad = np.deg2rad(theta_deg)
    cos_val = np.cos(rad)
    cos_val = np.clip(cos_val, 1e-3, None)
    return 1.0 / cos_val


def f_grit_m(grit: float, p: ModelParameters) -> float:
    return (p.grit_ref / grit) ** p.n_grit_m


def f_grit_T(grit: float, p: ModelParameters) -> float:
    return (p.grit_ref / grit) ** p.n_grit_T


def f_grit_R(grit: float, p: ModelParameters) -> float:
    return (grit / p.grit_ref) ** p.n_grit_R


def f_wood_m(hardness: float) -> float:
    hardness = np.clip(hardness, 1e-3, None)
    return 1.0 / hardness


def f_wood_T(hardness: float, p: ModelParameters) -> float:
    hardness = np.clip(hardness, 1e-3, None)
    return hardness ** p.n_hard_T


def f_wood_R(hardness: float, p: ModelParameters) -> float:
    hardness = np.clip(hardness, 1e-3, None)
    return hardness ** p.n_hard_R


def plywood_sanding_ode(t: float, x: np.ndarray, u: Inputs, p: ModelParameters) -> List[float]:
    """Four-state ODE describing plywood edge sanding."""

    z, T, W, Ra = x

    dzdt = (
        p.k_m
        * u.pressure ** p.a_p
        * u.v_rel ** p.a_v
        * f_theta_m(u.theta_deg)
        * f_grit_m(u.grit, p)
        * f_wood_m(u.hardness)
        * (1.0 - W)
    )

    dTdt = (
        p.k_T
        * u.pressure
        * u.v_rel
        * f_theta_m(u.theta_deg)
        * f_grit_T(u.grit, p)
        * f_wood_T(u.hardness, p)
        * (1.0 - W)
        - (T - p.T_env) / p.tau_T
    )

    dWdt = p.k_w * abs(dzdt)
    if W >= 1.0 and dWdt > 0.0:
        dWdt = 0.0

    Ra_ss = (
        p.k_R
        * f_grit_R(u.grit, p)
        * f_theta_m(u.theta_deg)
        * f_wood_R(u.hardness, p)
        * (1.0 + p.gamma_W * W)
    )
    dRadt = -(Ra - Ra_ss) / p.tau_R

    return [dzdt, dTdt, dWdt, dRadt]


def get_default_parameters() -> ModelParameters:
    return ModelParameters()


def run_simulation(
    u: Inputs,
    p: ModelParameters,
    x0: Iterable[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the sanding model and return time vector and state history."""

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(
        lambda t, x: plywood_sanding_ode(t, x, u, p),
        t_span=t_span,
        y0=np.array(x0, dtype=float),
        t_eval=t_eval,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.t, sol.y


def run_scenario_grid(
    angles: Iterable[float],
    grits: Iterable[float],
    pressures: Iterable[float],
    hardness_values: Iterable[float],
    p: ModelParameters,
    t_span: Tuple[float, float] = (0.0, 5.0),
    x0: Iterable[float] = (0.0, 20.0, 0.0, 8.0),
) -> pd.DataFrame:
    """Run simulations for all scenario combinations and summarize results."""

    records: List[Dict[str, float]] = []
    for theta_deg, grit, pressure, hardness in itertools.product(
        angles, grits, pressures, hardness_values
    ):
        u = Inputs(theta_deg=theta_deg, pressure=pressure, v_rel=15.0, hardness=hardness, grit=grit)
        t, y = run_simulation(u, p, x0, t_span)
        z, T, W, Ra = y
        duration = t_span[1] - t_span[0]
        records.append(
            {
                "theta_deg": theta_deg,
                "grit": grit,
                "pressure": pressure,
                "hardness": hardness,
                "z_final": float(z[-1]),
                "Ra_final": float(Ra[-1]),
                "T_max": float(T.max()),
                "W_final": float(min(W[-1], 1.0)),
                "mrr_avg": float(z[-1] / duration),
            }
        )

    return pd.DataFrame.from_records(records)


def plot_time_series(t: np.ndarray, y: np.ndarray, title: str = "Sanding simulation", save_path: str | None = None):
    """Plot time evolution of all state variables."""

    labels = ["z [mm]", "T [°C]", "W [-]", "Ra [µm]"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()
    for ax, data, label in zip(axes, y, labels):
        ax.plot(t, data, lw=2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time [s]")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axes


def plot_grid_summary(df: pd.DataFrame, value_col: str, title: str, save_path: str | None = None):
    """Visualize aggregated metric as grouped bars."""

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = df.groupby(["theta_deg", "grit"])[value_col].mean().unstack()
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Grit")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


benchmark_data = [
    {
        "name": "birch_plywood_P120_45deg_medium_pressure",
        "theta_deg": 45,
        "grit": 120,
        "pressure": 0.8,
        "v_rel": 15.0,
        "hardness": 0.7,
        "expected_Ra_range": (3.0, 5.0),
        "expected_MRR_range": (0.05, 0.2),
    },
    {
        "name": "softwood_P80_0deg_low_pressure",
        "theta_deg": 0,
        "grit": 80,
        "pressure": 0.3,
        "v_rel": 15.0,
        "hardness": 0.4,
        "expected_Ra_range": (4.0, 7.0),
        "expected_MRR_range": (0.06, 0.18),
    },
]


def compare_with_benchmarks(sim_results_df: pd.DataFrame, benchmarks: List[Dict]) -> pd.DataFrame:
    """Compare simulation outputs with simple benchmark ranges."""

    reports: List[Dict[str, object]] = []
    for bench in benchmarks:
        mask = (
            (sim_results_df["theta_deg"] == bench["theta_deg"]) &
            (sim_results_df["grit"] == bench["grit"]) &
            (sim_results_df["pressure"] == bench["pressure"]) &
            (sim_results_df["hardness"] == bench["hardness"])
        )
        matches = sim_results_df[mask]
        if matches.empty:
            reports.append({"name": bench["name"], "status": "no_match"})
            continue

        row = matches.iloc[0]
        z_rate = row["mrr_avg"]
        Ra_final = row["Ra_final"]
        mrr_min, mrr_max = bench["expected_MRR_range"]
        ra_min, ra_max = bench["expected_Ra_range"]
        reports.append(
            {
                "name": bench["name"],
                "mrr_within": mrr_min <= z_rate <= mrr_max,
                "Ra_within": ra_min <= Ra_final <= ra_max,
                "mrr_avg": z_rate,
                "Ra_final": Ra_final,
                "z_final": row["z_final"],
                "T_max": row["T_max"],
                "W_final": row["W_final"],
            }
        )

    return pd.DataFrame(reports)


def demo_single_run():
    """Run a representative single simulation and produce plots."""

    p = get_default_parameters()
    u = Inputs(theta_deg=45.0, pressure=0.8, v_rel=15.0, hardness=0.7, grit=120.0)
    t_span = (0.0, 5.0)
    x0 = (0.0, p.T_env, 0.0, 8.0)
    t, y = run_simulation(u, p, x0, t_span)
    plot_time_series(t, y, title="Single sanding scenario", save_path="single_run.png")
    print("Saved single scenario plot to single_run.png")


def demo_grid_and_benchmark():
    """Run a grid sweep and benchmark comparison."""

    p = get_default_parameters()
    df = run_scenario_grid(
        angles=[0, 45, 90],
        grits=[80, 120, 180],
        pressures=[0.3, 0.8, 1.5],
        hardness_values=[0.4, 0.7, 0.9],
        p=p,
    )
    print(df.head())
    df.to_csv("scenario_summary.csv", index=False)
    print("Saved scenario summary to scenario_summary.csv")
    plot_grid_summary(df, value_col="Ra_final", title="Final Ra vs angle and grit", save_path="grid_ra.png")
    print("Saved grid summary plot to grid_ra.png")

    bench_report = compare_with_benchmarks(df, benchmark_data)
    print("Benchmark comparison:")
    print(bench_report)
    bench_report.to_csv("benchmark_report.csv", index=False)


if __name__ == "__main__":
    demo_single_run()
    demo_grid_and_benchmark()
