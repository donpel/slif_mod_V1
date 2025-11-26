"""
Self-contained sanding roughness model script.
Implements dynamic ODEs for plywood belt sanding with parameter sweeps.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class Inputs:
    theta_deg: float
    FN: float
    v_rel: float
    hardness: float
    grit: int


@dataclass
class ModelParameters:
    # Material removal
    k_m: float
    a_FN: float
    a_v: float

    # Temperature
    k_T: float
    T_env: float
    tau_T: float

    # Wear
    k_w0: float
    alpha_T: float
    W_max: float

    # Roughness
    k_R0: float
    gamma_W_Ra: float
    gamma_MRR_Ra: float
    tau_R: float

    # Grit and hardness influences
    grit_ref: int
    n_grit_m: float
    n_grit_R: float
    n_hard_m: float
    n_hard_R: float


def get_default_parameters() -> ModelParameters:
    """Return a set of plausible default parameters for the sanding model."""
    return ModelParameters(
        k_m=2.5e-5,  # mm/(s * N^a * (m/s)^b)
        a_FN=1.0,
        a_v=1.0,
        k_T=0.06,  # temperature rise rate coefficient
        T_env=25.0,
        tau_T=25.0,
        k_w0=2.0e-4,
        alpha_T=0.015,
        W_max=1.0,
        k_R0=6.0,  # base roughness in microns
        gamma_W_Ra=1.3,
        gamma_MRR_Ra=0.45,
        tau_R=10.0,
        grit_ref=120,
        n_grit_m=1.1,
        n_grit_R=1.2,
        n_hard_m=1.3,
        n_hard_R=0.6,
    )


def f_theta(theta_deg: float) -> float:
    """Angle effect: larger angles (closer to 90 deg) increase effective aggressiveness."""
    theta_rad = np.deg2rad(theta_deg)
    return 1.0 / max(np.cos(theta_rad), 1e-3)


def f_grit_m(grit: int, p: ModelParameters) -> float:
    """Coarser grit (smaller number) increases removal rate via a power-law."""
    return (p.grit_ref / grit) ** p.n_grit_m


def f_grit_R(grit: int, p: ModelParameters) -> float:
    """Finer grit (larger number) reduces roughness via a power-law."""
    return (grit / p.grit_ref) ** p.n_grit_R


def f_hard_m(hardness: float, p: ModelParameters) -> float:
    """Harder material reduces removal rate."""
    return hardness ** (-p.n_hard_m)


def f_hard_R(hardness: float, p: ModelParameters) -> float:
    """Harder material can slightly increase resulting roughness."""
    return hardness ** (p.n_hard_R)


def sanding_ode(t: float, x: np.ndarray, u: Inputs, p: ModelParameters):
    """ODE system for sanding state evolution."""
    z, T, W, Ra = x

    # Material removal rate [mm/s]
    dzdt = (
        p.k_m
        * (u.FN ** p.a_FN)
        * (u.v_rel ** p.a_v)
        * f_theta(u.theta_deg)
        * f_grit_m(u.grit, p)
        * f_hard_m(u.hardness, p)
        * (1.0 - W)
    )

    # Temperature dynamics [C/s]
    dTdt = p.k_T * u.FN * u.v_rel * (1.0 - W) - (T - p.T_env) / p.tau_T

    # Wear dynamics [1/s]
    wear_base = p.k_w0 * u.FN * u.v_rel
    wear_rate = wear_base * (1.0 + p.alpha_T * max(T - p.T_env, 0.0) / max(p.T_env, 1e-3))
    dWdt = wear_rate
    if W >= p.W_max and dWdt > 0:
        dWdt = 0.0

    # Roughness dynamics [microns/s]
    normalized_aggressiveness = dzdt / max(p.k_m * (20 ** p.a_FN) * (15 ** p.a_v), 1e-9)
    Ra_ss = (
        p.k_R0
        * f_grit_R(u.grit, p)
        * f_hard_R(u.hardness, p)
        * (1.0 + p.gamma_W_Ra * W)
        * (1.0 + p.gamma_MRR_Ra * normalized_aggressiveness)
    )
    dRadt = -(Ra - Ra_ss) / p.tau_R

    return [dzdt, dTdt, dWdt, dRadt]


def run_simulation(
    u: Inputs,
    p: ModelParameters,
    t_span: tuple[float, float],
    x0,
    n_points: int = 300,
):
    """Integrate the sanding ODE over t_span and return time and state history."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        sanding_ode,
        t_span,
        x0,
        t_eval=t_eval,
        args=(u, p),
        method="RK45",
        vectorized=False,
    )
    x = sol.y
    x[2, :] = np.clip(x[2, :], 0.0, p.W_max)
    return sol.t, x


def plot_time_series(t: np.ndarray, x: np.ndarray, title: str, filename: str):
    """Plot z, T, W, Ra time series as a 2x2 grid and save to filename."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, x[0])
    axes[0, 0].set_title("Material removal z")
    axes[0, 0].set_ylabel("z [mm]")
    axes[0, 1].plot(t, x[1], color="orange")
    axes[0, 1].set_title("Contact temperature T")
    axes[0, 1].set_ylabel("T [°C]")
    axes[1, 0].plot(t, x[2], color="green")
    axes[1, 0].set_title("Abrasive wear W")
    axes[1, 0].set_ylabel("W [-]")
    axes[1, 1].plot(t, x[3], color="red")
    axes[1, 1].set_title("Surface roughness Ra")
    axes[1, 1].set_ylabel("Ra [µm]")
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def sweep_speed_forces(p: ModelParameters) -> pd.DataFrame:
    """Run simulations over grids of FN and v_rel; return summary DataFrame."""
    FN_values = [5, 10, 20, 40]
    v_values = [5, 10, 15, 20]
    theta_deg = 90.0
    hardness = 0.7
    grit = 180
    t_span = (0.0, 60.0)
    x0 = [0.0, p.T_env, 0.0, 8.0]

    records = []
    for FN in FN_values:
        for v_rel in v_values:
            u = Inputs(theta_deg=theta_deg, FN=FN, v_rel=v_rel, hardness=hardness, grit=grit)
            t, x = run_simulation(u, p, t_span, x0)
            z_final = float(x[0, -1])
            W_final = float(x[2, -1])
            Ra_final = float(x[3, -1])
            T_max = float(np.max(x[1]))
            MRR_avg = z_final / (t_span[1] - t_span[0])
            records.append(
                {
                    "FN": FN,
                    "v_rel": v_rel,
                    "z_final_mm": z_final,
                    "W_final": W_final,
                    "Ra_final": Ra_final,
                    "T_max": T_max,
                    "MRR_avg_mm_s": MRR_avg,
                }
            )
    df = pd.DataFrame.from_records(records)
    return df


def plot_sweep_results(df: pd.DataFrame):
    """Generate summary plots for sweep results and save PNG files."""
    # Ra vs speed for each FN
    fig, ax = plt.subplots(figsize=(8, 6))
    for FN, group in df.groupby("FN"):
        ax.plot(group["v_rel"], group["Ra_final"], marker="o", label=f"FN={FN} N")
    ax.set_xlabel("Belt speed v_rel [m/s]")
    ax.set_ylabel("Final roughness Ra [µm]")
    ax.set_title("Ra vs belt speed for different normal forces")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Ra_vs_speed_forces.png", dpi=200)
    plt.close(fig)

    # Wear vs speed for each FN
    fig, ax = plt.subplots(figsize=(8, 6))
    for FN, group in df.groupby("FN"):
        ax.plot(group["v_rel"], group["W_final"], marker="s", label=f"FN={FN} N")
    ax.set_xlabel("Belt speed v_rel [m/s]")
    ax.set_ylabel("Final wear W [-]")
    ax.set_title("Abrasive wear vs belt speed")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Wear_vs_speed_forces.png", dpi=200)
    plt.close(fig)

    # Ra vs FN for each speed
    fig, ax = plt.subplots(figsize=(8, 6))
    for v_rel, group in df.groupby("v_rel"):
        ax.plot(group["FN"], group["Ra_final"], marker="^", label=f"v_rel={v_rel} m/s")
    ax.set_xlabel("Normal force FN [N]")
    ax.set_ylabel("Final roughness Ra [µm]")
    ax.set_title("Ra vs normal force for different speeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Ra_vs_FN_speeds.png", dpi=200)
    plt.close(fig)

    # Heatmap for Ra
    pivot_ra = df.pivot(index="FN", columns="v_rel", values="Ra_final")
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(pivot_ra.values, origin="lower", aspect="auto",
                  extent=[pivot_ra.columns.min(), pivot_ra.columns.max(), pivot_ra.index.min(), pivot_ra.index.max()],
                  cmap="inferno")
    ax.set_xlabel("Belt speed v_rel [m/s]")
    ax.set_ylabel("Normal force FN [N]")
    ax.set_title("Heatmap of Ra_final")
    fig.colorbar(c, ax=ax, label="Ra [µm]")
    fig.tight_layout()
    fig.savefig("Ra_heatmap.png", dpi=200)
    plt.close(fig)

    # Heatmap for wear
    pivot_w = df.pivot(index="FN", columns="v_rel", values="W_final")
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(pivot_w.values, origin="lower", aspect="auto",
                  extent=[pivot_w.columns.min(), pivot_w.columns.max(), pivot_w.index.min(), pivot_w.index.max()],
                  cmap="viridis")
    ax.set_xlabel("Belt speed v_rel [m/s]")
    ax.set_ylabel("Normal force FN [N]")
    ax.set_title("Heatmap of W_final")
    fig.colorbar(c, ax=ax, label="W [-]")
    fig.tight_layout()
    fig.savefig("Wear_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    p = get_default_parameters()

    # Baseline scenario
    u_base = Inputs(theta_deg=90.0, FN=20.0, v_rel=15.0, hardness=0.7, grit=180)
    t_span = (0.0, 60.0)
    x0 = [0.0, p.T_env, 0.0, 8.0]

    t, x = run_simulation(u_base, p, t_span, x0)
    plot_time_series(t, x, "Baseline sanding dynamics", "baseline_time_series.png")

    print("Baseline results (60 s):")
    print(f"  Removed thickness z_final = {x[0, -1]:.4f} mm")
    print(f"  Final wear W_final = {x[2, -1]:.3f}")
    print(f"  Final roughness Ra_final = {x[3, -1]:.2f} µm")
    print(f"  Max temperature T_max = {np.max(x[1]):.2f} °C")

    # Sweep
    df_sweep = sweep_speed_forces(p)
    print("\nSweep results (head):")
    print(df_sweep.head())
    df_sweep.to_csv("sweep_results.csv", index=False)
    plot_sweep_results(df_sweep)

    # Optional comparison for bevel edge
    u_bevel = Inputs(theta_deg=45.0, FN=20.0, v_rel=15.0, hardness=0.7, grit=180)
    t_bevel, x_bevel = run_simulation(u_bevel, p, t_span, x0)
    plot_time_series(t_bevel, x_bevel, "Bevel edge dynamics (45 deg)", "bevel_time_series.png")
    print("\nBevel edge roughness after 60 s: {:.2f} µm".format(x_bevel[3, -1]))


if __name__ == "__main__":
    main()
