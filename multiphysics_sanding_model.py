import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def contact_area_mm2(theta_deg: float, h_edge_mm: float, L_contact_mm: float) -> float:
    """Compute contact area A_c in mm^2 using projected edge height and contact length."""
    theta_rad = np.deg2rad(theta_deg)
    cos_t = max(np.cos(theta_rad), 1e-3)
    return (h_edge_mm / cos_t) * L_contact_mm


def angle_factor(theta_deg: float, p: "ModelParameters") -> float:
    """Angle factor used for material removal scaling."""
    theta_rad = np.deg2rad(theta_deg)
    cos_t = max(np.cos(theta_rad), 1e-3)
    return cos_t ** (-p.n_theta_m)


def f_grit_m(grit: int, p: "ModelParameters") -> float:
    """Coarser grit (smaller grit number) increases aggressiveness."""
    return (p.grit_ref / max(grit, 1e-6)) ** p.n_grit_m


def f_grit_R(grit: int, p: "ModelParameters") -> float:
    """Finer grit (higher number) tends to reduce cutting-related roughness."""
    return (max(grit, 1e-6) / p.grit_ref) ** p.n_grit_R


def f_hard_m(hardness: float, p: "ModelParameters") -> float:
    return hardness ** (-p.n_hard_m)


def f_hard_R(hardness: float, p: "ModelParameters") -> float:
    return hardness ** p.n_hard_R


def effective_MRR_factor(W_mech: float, W_load: float, p: "ModelParameters") -> float:
    factor = (1.0 - p.alpha_W_mech * W_mech) * (1.0 - p.alpha_W_load * W_load)
    return float(np.clip(factor, 0.0, 1.0))


@dataclass
class Inputs:
    theta_deg: float
    FN: float
    v_rel: float
    hardness: float
    grit: int


@dataclass
class ModelParameters:
    # Contact & removal
    k_m: float
    H_eff: float
    alpha_W_mech: float
    alpha_W_load: float
    n_theta_m: float
    grit_ref: float
    n_grit_m: float
    n_hard_m: float

    # Friction and heat
    mu0: float
    eta_heat: float
    rho_wood: float
    c_wood: float
    h_eff_mm: float
    h_conv: float
    T_env: float

    # Wear
    K_mech: float
    K_load0: float
    beta_T_load: float
    W_mech_max: float
    W_load_max: float

    # Roughness
    k_R_cut: float
    k_R_damage: float
    gamma_W_Ra: float
    tau_R: float
    n_grit_R: float
    n_hard_R: float


def get_default_parameters() -> ModelParameters:
    """Return plausible default parameters for plywood sanding."""
    return ModelParameters(
        k_m=0.15,  # base removal coefficient
        H_eff=60e6,  # Pa effective hardness
        alpha_W_mech=0.7,
        alpha_W_load=0.9,
        n_theta_m=1.0,
        grit_ref=120.0,
        n_grit_m=0.8,
        n_hard_m=0.6,
        mu0=0.6,
        eta_heat=0.6,
        rho_wood=600.0,  # kg/m^3
        c_wood=1700.0,  # J/(kg*K)
        h_eff_mm=1.5,
        h_conv=35.0,
        T_env=25.0,
        K_mech=1.5e-5,
        K_load0=0.08,
        beta_T_load=0.015,
        W_mech_max=1.0,
        W_load_max=1.0,
        k_R_cut=4.5,
        k_R_damage=2.5,
        gamma_W_Ra=1.8,
        tau_R=8.0,
        n_grit_R=0.9,
        n_hard_R=0.3,
    )


def sanding_ode(t: float, x: np.ndarray, u: Inputs, p: ModelParameters) -> List[float]:
    z, T, W_mech, W_load, Ra = x

    # Contact geometry
    A_c_mm2 = contact_area_mm2(u.theta_deg, h_edge_mm=11.0, L_contact_mm=100.0)
    A_c_m2 = A_c_mm2 * 1e-6
    p_contact = u.FN / max(A_c_m2, 1e-12)

    # Material removal
    base_MRR = (
        p.k_m
        * (p_contact / max(p.H_eff, 1e-12))
        * u.v_rel
        * angle_factor(u.theta_deg, p)
        * f_grit_m(u.grit, p)
        * f_hard_m(u.hardness, p)
    )
    dzdt = (base_MRR / max(A_c_mm2, 1e-6)) * effective_MRR_factor(W_mech, W_load, p) * 1e3

    # Frictional heating
    mu_eff = p.mu0
    friction_power = mu_eff * p_contact * A_c_m2 * u.v_rel
    heat_to_wood = p.eta_heat * friction_power
    q_wood = heat_to_wood / max(A_c_m2, 1e-12)

    h_eff_m = p.h_eff_mm * 1e-3
    dTdt = (q_wood - p.h_conv * (T - p.T_env)) / max(p.rho_wood * p.c_wood * h_eff_m, 1e-9)

    # Mechanical wear
    dW_mech_dt = p.K_mech * (p_contact * u.v_rel) / max(p.H_eff, 1e-12)

    # Loading/clogging wear
    load_base = p.K_load0 * max(dzdt, 0.0)
    temp_factor = np.exp(p.beta_T_load * max(T - p.T_env, 0.0))
    dW_load_dt = load_base * temp_factor

    # Roughness dynamics
    Ra_cut = p.k_R_cut * f_grit_R(u.grit, p) * f_hard_R(u.hardness, p)
    W_total = 0.5 * (W_mech + W_load)
    Ra_ss = Ra_cut + p.k_R_damage * (1.0 + p.gamma_W_Ra * W_total)
    dRadt = -(Ra - Ra_ss) / p.tau_R

    return [dzdt, dTdt, dW_mech_dt, dW_load_dt, dRadt]


def run_simulation(u: Inputs, p: ModelParameters, t_span: Tuple[float, float], x0, n_points: int = 300):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        sanding_ode,
        t_span,
        x0,
        t_eval=t_eval,
        method="RK45",
        args=(u, p),
        vectorized=False,
    )
    x = sol.y
    x[2, :] = np.clip(x[2, :], 0.0, p.W_mech_max)
    x[3, :] = np.clip(x[3, :], 0.0, p.W_load_max)
    return sol.t, x


def plot_time_series(t: np.ndarray, x: np.ndarray, title: str, filename: str) -> None:
    z, T, W_mech, W_load, Ra = x
    W_total = 0.5 * (W_mech + W_load)
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    ax_list = axs.flatten()

    ax_list[0].plot(t, z)
    ax_list[0].set_ylabel("Removed thickness z [mm]")

    ax_list[1].plot(t, T, color="tab:red")
    ax_list[1].set_ylabel("Temperature [°C]")

    ax_list[2].plot(t, W_mech, label="Mechanical")
    ax_list[2].set_ylabel("W_mech [-]")
    ax_list[2].legend()

    ax_list[3].plot(t, W_load, label="Loading", color="tab:purple")
    ax_list[3].set_ylabel("W_load [-]")
    ax_list[3].legend()

    ax_list[4].plot(t, Ra, color="tab:green")
    ax_list[4].set_ylabel("Ra [µm]")
    ax_list[4].set_xlabel("Time [s]")

    ax_list[5].plot(t, W_total, color="tab:orange")
    ax_list[5].set_ylabel("W_total [-]")
    ax_list[5].set_xlabel("Time [s]")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def plot_sweep_results(df: pd.DataFrame, prefix: str) -> None:
    # Ra vs speed for each FN
    plt.figure(figsize=(8, 5))
    for fn, group in df.groupby("FN"):
        plt.plot(group["v_rel"], group["Ra_final"], marker="o", label=f"FN={fn} N")
    plt.xlabel("Belt speed v_rel [m/s]")
    plt.ylabel("Final Ra [µm]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_Ra_vs_speed_forces.png", dpi=200)
    plt.close()

    # W_total vs speed
    plt.figure(figsize=(8, 5))
    for fn, group in df.groupby("FN"):
        plt.plot(group["v_rel"], group["W_total_final"], marker="s", label=f"FN={fn} N")
    plt.xlabel("Belt speed v_rel [m/s]")
    plt.ylabel("Final W_total [-]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_Wtotal_vs_speed_forces.png", dpi=200)
    plt.close()

    # Ra vs FN for each speed
    plt.figure(figsize=(8, 5))
    for v_rel, group in df.groupby("v_rel"):
        plt.plot(group["FN"], group["Ra_final"], marker="^", label=f"v={v_rel} m/s")
    plt.xlabel("Normal force FN [N]")
    plt.ylabel("Final Ra [µm]")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_Ra_vs_FN_speeds.png", dpi=200)
    plt.close()

    # Heatmaps
    pivot_ra = df.pivot(index="FN", columns="v_rel", values="Ra_final")
    pivot_w = df.pivot(index="FN", columns="v_rel", values="W_total_final")

    for data, name, cmap in [
        (pivot_ra, "Ra_heatmap", "viridis"),
        (pivot_w, "Wtotal_heatmap", "magma"),
    ]:
        plt.figure(figsize=(6, 5))
        c = plt.imshow(data.values, aspect="auto", origin="lower", cmap=cmap,
                       extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        plt.colorbar(c, label="Ra [µm]" if "Ra" in name else "W_total [-]")
        plt.xlabel("Belt speed v_rel [m/s]")
        plt.ylabel("Normal force FN [N]")
        plt.title(name)
        plt.savefig(f"{prefix}_{name}.png", dpi=200)
        plt.close()


def sweep_speed_forces(
    p: ModelParameters,
    theta_deg: float = 90.0,
    hardness: float = 0.7,
    grit: int = 180,
    t_span: Tuple[float, float] = (0.0, 60.0),
) -> pd.DataFrame:
    FN_values = [5, 10, 20, 40]
    v_values = [5, 10, 15, 20]

    records = []
    for FN in FN_values:
        for v_rel in v_values:
            u = Inputs(theta_deg=theta_deg, FN=FN, v_rel=v_rel, hardness=hardness, grit=grit)
            x0 = [0.0, p.T_env, 0.0, 0.0, 8.0]
            t, x = run_simulation(u, p, t_span, x0)
            z_final = float(x[0, -1])
            T_max = float(np.max(x[1, :]))
            W_mech_final = float(x[2, -1])
            W_load_final = float(x[3, -1])
            W_total_final = 0.5 * (W_mech_final + W_load_final)
            Ra_final = float(x[4, -1])
            MRR_avg = z_final / max(t_span[1] - t_span[0], 1e-9)
            records.append(
                {
                    "FN": FN,
                    "v_rel": v_rel,
                    "theta_deg": theta_deg,
                    "z_final_mm": z_final,
                    "T_max": T_max,
                    "W_mech_final": W_mech_final,
                    "W_load_final": W_load_final,
                    "W_total_final": W_total_final,
                    "Ra_final": Ra_final,
                    "MRR_avg_mm_s": MRR_avg,
                }
            )
    return pd.DataFrame(records)


def compare_angles(df90: pd.DataFrame, df45: pd.DataFrame) -> pd.DataFrame:
    merged = df90.merge(df45, on=["FN", "v_rel"], suffixes=("_90", "_45"))
    merged["Ra_diff_45_minus_90"] = merged["Ra_final_45"] - merged["Ra_final_90"]
    return merged[["FN", "v_rel", "Ra_final_90", "Ra_final_45", "Ra_diff_45_minus_90"]]


if __name__ == "__main__":
    p = get_default_parameters()

    u_base = Inputs(theta_deg=90.0, FN=20.0, v_rel=15.0, hardness=0.7, grit=180)
    t_span = (0.0, 60.0)
    x0 = [0.0, p.T_env, 0.0, 0.0, 8.0]
    t, x = run_simulation(u_base, p, t_span, x0)
    plot_time_series(t, x, "Baseline sanding (90 deg)", "baseline_time_series_90deg.png")

    print("Baseline summary (90 deg):")
    print(f"  z_final [mm]: {x[0, -1]:.4f}")
    print(f"  T_max   [°C]: {np.max(x[1, :]):.2f}")
    print(f"  W_mech_final: {x[2, -1]:.3f}")
    print(f"  W_load_final: {x[3, -1]:.3f}")
    print(f"  Ra_final [µm]: {x[4, -1]:.3f}")

    df_90 = sweep_speed_forces(p, theta_deg=90.0)
    print("Sweep results for 90° (head):")
    print(df_90.head())
    df_90.to_csv("sweep_results_90deg.csv", index=False)
    plot_sweep_results(df_90, prefix="theta90")

    df_45 = sweep_speed_forces(p, theta_deg=45.0)
    print("Sweep results for 45° (head):")
    print(df_45.head())
    df_45.to_csv("sweep_results_45deg.csv", index=False)
    plot_sweep_results(df_45, prefix="theta45")

    comparison = compare_angles(df_90, df_45)
    print("Ra comparison (45° - 90°):")
    print(comparison.head())
