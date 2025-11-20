import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# -----------------------------
# Dataclass definitions
# -----------------------------

@dataclass
class ModelParameters:
    """Container for model parameters with initial defaults that can be fitted later."""
    k_m: float = 0.02
    a_FN: float = 1.0
    a_v: float = 1.0
    k_T: float = 0.08
    T_env: float = 20.0
    tau_T: float = 2.5
    k_w: float = 0.02
    k_R: float = 4.0
    gamma_W: float = 0.4
    tau_R: float = 1.5
    grit_ref: float = 180.0
    n_grit_m: float = 0.25
    n_grit_T: float = 0.1
    n_grit_R: float = 0.35
    n_hard_T: float = 0.2
    n_hard_R: float = 0.25

@dataclass
class Inputs:
    """Inputs to the sanding process for a single simulation."""
    theta_deg: float
    FN: float
    v_rel: float
    hardness: float
    grit: float

# -----------------------------
# Helper functions
# -----------------------------


def f_theta_m(theta_deg: float) -> float:
    """Angle effect on removal and roughness; avoid division by zero near 90 deg."""
    theta_rad = math.radians(theta_deg)
    denom = max(math.cos(theta_rad), 1e-3)
    return 1.0 / denom


def f_grit_m(grit: float, p: ModelParameters) -> float:
    grit_safe = max(grit, 1e-3)
    return (p.grit_ref / grit_safe) ** p.n_grit_m


def f_grit_T(grit: float, p: ModelParameters) -> float:
    grit_safe = max(grit, 1e-3)
    return (p.grit_ref / grit_safe) ** p.n_grit_T


def f_grit_R(grit: float, p: ModelParameters) -> float:
    grit_safe = max(grit, 1e-3)
    return (grit_safe / p.grit_ref) ** p.n_grit_R


def f_wood_m(hardness: float) -> float:
    hardness_safe = max(hardness, 1e-3)
    return 1.0 / hardness_safe


def f_wood_T(hardness: float, p: ModelParameters) -> float:
    hardness_safe = max(hardness, 1e-3)
    return hardness_safe ** p.n_hard_T


def f_wood_R(hardness: float, p: ModelParameters) -> float:
    hardness_safe = max(hardness, 1e-3)
    return hardness_safe ** p.n_hard_R

# -----------------------------
# Core ODE
# -----------------------------


def plywood_sanding_ode(t: float, x: np.ndarray, u: Inputs, p: ModelParameters) -> List[float]:
    """Four-state ODE representing local sanding dynamics."""
    z, T, W, Ra = x

    theta_factor = f_theta_m(u.theta_deg)
    grit_m = f_grit_m(u.grit, p)
    grit_T = f_grit_T(u.grit, p)
    grit_R = f_grit_R(u.grit, p)
    wood_m = f_wood_m(u.hardness)
    wood_T = f_wood_T(u.hardness, p)
    wood_R = f_wood_R(u.hardness, p)

    dzdt = (
        p.k_m
        * (u.FN ** p.a_FN)
        * (u.v_rel ** p.a_v)
        * theta_factor
        * grit_m
        * wood_m
        * (1.0 - W)
    )

    dTdt = (
        p.k_T
        * u.FN
        * u.v_rel
        * theta_factor
        * grit_T
        * wood_T
        * (1.0 - W)
        - (T - p.T_env) / p.tau_T
    )

    dWdt = p.k_w * abs(dzdt)
    if W >= 1.0 and dWdt > 0:
        dWdt = 0.0

    Ra_ss = (
        p.k_R
        * grit_R
        * theta_factor
        * wood_R
        * (1.0 + p.gamma_W * W)
    )
    dRadt = -(Ra - Ra_ss) / p.tau_R

    return [dzdt, dTdt, dWdt, dRadt]

# -----------------------------
# Parameter factory
# -----------------------------


def get_default_parameters() -> ModelParameters:
    """Return default parameters before data-driven fitting."""
    return ModelParameters()

# -----------------------------
# Experimental data handling
# -----------------------------


def create_experiment_dataframe() -> pd.DataFrame:
    """Create DataFrame from hard-coded experimental sanding dataset."""
    data = [
        {"weight_g": 451, "time_s": 54, "Ft_N": 1.578082192, "FN_N": 4.42431, "speed_mm_min": 5.555555556},
        {"weight_g": 1100, "time_s": 28, "Ft_N": 6.575342466, "FN_N": 10.791, "speed_mm_min": 10.71428571},
        {"weight_g": 1400, "time_s": 23, "Ft_N": 8.153424658, "FN_N": 13.734, "speed_mm_min": 13.04347826},
        {"weight_g": 1915, "time_s": 16, "Ft_N": 9.468493151, "FN_N": 18.78615, "speed_mm_min": 18.75},
        {"weight_g": 2212, "time_s": 13, "Ft_N": 14.72876712, "FN_N": 21.69972, "speed_mm_min": 23.07692308},
        {"weight_g": 5000, "time_s": 6, "Ft_N": 29.19452055, "FN_N": 49.05, "speed_mm_min": 50.0},
    ]
    df = pd.DataFrame(data)
    df["MRR_mm_per_min"] = df["speed_mm_min"]
    df["MRR_mm_per_s"] = df["MRR_mm_per_min"] / 60.0
    df["ratio_Ft_to_FN"] = df["Ft_N"] / df["FN_N"]
    return df

# -----------------------------
# Fitting utilities
# -----------------------------


def fit_mrr_regression(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Fit linear MRR(FN) regression and return coefficients and R^2."""
    x = df["FN_N"].values
    y = df["MRR_mm_per_min"].values
    coeffs = np.polyfit(x, y, 1)
    a, b = coeffs
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def _ode_mrr_model(FN_array: np.ndarray, k_m: float, a_FN: float, a_v: float, p: ModelParameters, v_rel: float, hardness: float, grit: float) -> np.ndarray:
    """Instantaneous MRR prediction (mm/min) at W=0 for fitting."""
    theta_factor = f_theta_m(90.0)
    grit_m = f_grit_m(grit, p)
    wood_m = f_wood_m(hardness)
    dzdt_mm_s = k_m * (FN_array ** a_FN) * (v_rel ** a_v) * theta_factor * grit_m * wood_m
    return dzdt_mm_s * 60.0


def fit_material_removal_parameters(df: pd.DataFrame, p: ModelParameters, v_rel_fit: float, hardness_fit: float, grit_fit: float) -> Tuple[ModelParameters, Dict[str, float]]:
    """Fit k_m, a_FN, a_v so ODE removal matches experimental MRR for given conditions."""
    FN_data = df["FN_N"].values
    MRR_data = df["MRR_mm_per_min"].values

    def model(FN_array, k_m, a_FN, a_v):
        return _ode_mrr_model(FN_array, k_m, a_FN, a_v, p, v_rel_fit, hardness_fit, grit_fit)

    initial_guess = [p.k_m, p.a_FN, p.a_v]
    bounds = ([1e-5, 0.1, 0.1], [10.0, 3.0, 3.0])
    popt, pcov = curve_fit(model, FN_data, MRR_data, p0=initial_guess, bounds=bounds, maxfev=20000)
    p.k_m, p.a_FN, p.a_v = popt

    MRR_pred = model(FN_data, *popt)
    ss_res = np.sum((MRR_data - MRR_pred) ** 2)
    ss_tot = np.sum((MRR_data - np.mean(MRR_data)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    diagnostics = {
        "k_m": p.k_m,
        "a_FN": p.a_FN,
        "a_v": p.a_v,
        "R2": r2,
        "MAE": float(np.mean(np.abs(MRR_data - MRR_pred))),
        "RMSE": float(np.sqrt(np.mean((MRR_data - MRR_pred) ** 2))),
    }
    return p, diagnostics

# -----------------------------
# Validation utilities
# -----------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and R2 metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def validate_mrr_regression(df: pd.DataFrame, a: float, b: float) -> Dict[str, float]:
    """Validate linear MRR regression vs experimental data."""
    y_true = df["MRR_mm_per_min"].values
    y_pred = a * df["FN_N"].values + b
    metrics = compute_metrics(y_true, y_pred)
    print("MRR regression validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def validate_material_removal_law(df: pd.DataFrame, p: ModelParameters, v_rel_fit: float, hardness_fit: float, grit_fit: float) -> Dict[str, float]:
    """Validate fitted ODE material removal law vs experimental MRR data."""
    FN_data = df["FN_N"].values
    MRR_true = df["MRR_mm_per_min"].values
    MRR_pred = _ode_mrr_model(FN_data, p.k_m, p.a_FN, p.a_v, p, v_rel_fit, hardness_fit, grit_fit)
    metrics = compute_metrics(MRR_true, MRR_pred)
    print("Material removal law validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics

# -----------------------------
# Simulation utilities
# -----------------------------


def run_dynamic_simulation(u: Inputs, p: ModelParameters, x0: List[float], t_span: Tuple[float, float], t_eval: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the plywood sanding ODE for given inputs and parameters."""
    sol = solve_ivp(
        fun=lambda t, x: plywood_sanding_ode(t, x, u, p),
        t_span=t_span,
        y0=x0,
        dense_output=False,
        t_eval=t_eval,
        max_step=0.05,
    )
    return sol.t, sol.y


def simulate_edge_scenario(
    FN_sim_values: List[float],
    t_s: float,
    p: ModelParameters,
    theta_deg: float,
    scenario_name: str = "edge_scenario",
    v_rel: float = 15.0,
    hardness: float = 0.7,
    grit: float = 180.0,
) -> pd.DataFrame:
    """Simulate sanding for multiple normal forces and compute removal and mass."""
    h_edge_mm = 11.0
    L_contact_mm = 100.0
    A_cm2 = (h_edge_mm / 10.0) * (L_contact_mm / 10.0)
    rho = 0.68  # g/cm^3

    records = []
    for FN_val in FN_sim_values:
        u = Inputs(theta_deg=theta_deg, FN=FN_val, v_rel=v_rel, hardness=hardness, grit=grit)
        x0 = [0.0, p.T_env, 0.0, 8.0]
        t_eval = np.linspace(0.0, t_s, 200)
        t, y = run_dynamic_simulation(u, p, x0, (0.0, t_s), t_eval=t_eval)
        z_final_mm = float(y[0, -1])
        T_final = float(y[1, -1])
        W_final = float(min(max(y[2, -1], 0.0), 1.0))
        Ra_final = float(y[3, -1])
        T_max = float(np.max(y[1, :]))

        z_final_cm = z_final_mm / 10.0
        V_cm3 = A_cm2 * z_final_cm
        m_g = V_cm3 * rho

        records.append(
            {
                "theta_deg": theta_deg,
                "FN": FN_val,
                "z_final_mm": z_final_mm,
                "V_cm3": V_cm3,
                "m_g": m_g,
                "T_final": T_final,
                "T_max": T_max,
                "W_final": W_final,
                "Ra_final": Ra_final,
                "scenario": scenario_name,
            }
        )

        plot_time_series(t, y, f"{scenario_name} FN={FN_val:.2f}N", f"{scenario_name}_FN_{FN_val:.2f}.png")

    df = pd.DataFrame(records)
    print(f"\nScenario '{scenario_name}' (theta={theta_deg} deg) summary:")
    print(df)
    return df

# -----------------------------
# Plotting utilities
# -----------------------------


def plot_time_series(t: np.ndarray, y: np.ndarray, title: str, filename: str) -> None:
    """Plot z, T, W, Ra vs time and save to file."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    labels = ["z (mm)", "T (°C)", "W (-)", "Ra (µm)"]
    for idx, ax in enumerate(axes):
        ax.plot(t, y[idx, :], label=labels[idx])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(labels[idx])
        ax.grid(True)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_mrr_vs_FN(df: pd.DataFrame, a: float, b: float, filename: str) -> None:
    """Plot experimental MRR vs FN with regression line."""
    plt.figure(figsize=(6, 4))
    x = df["FN_N"].values
    y = df["MRR_mm_per_min"].values
    plt.scatter(x, y, label="Experimental", color="blue")
    x_line = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, label=f"Fit: MRR = {a:.3f}*FN + {b:.3f}", color="red")
    plt.xlabel("Normal force FN (N)")
    plt.ylabel("MRR (mm/min)")
    plt.title("MRR vs Normal Force")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_validation_mrr(df: pd.DataFrame, p: ModelParameters, v_rel_fit: float, hardness_fit: float, grit_fit: float, filename: str) -> None:
    """Plot experimental vs ODE-based MRR predictions."""
    FN_data = df["FN_N"].values
    MRR_true = df["MRR_mm_per_min"].values
    MRR_pred = _ode_mrr_model(FN_data, p.k_m, p.a_FN, p.a_v, p, v_rel_fit, hardness_fit, grit_fit)

    plt.figure(figsize=(6, 4))
    plt.scatter(FN_data, MRR_true, label="Experimental", color="blue")
    plt.scatter(FN_data, MRR_pred, label="ODE prediction", color="green", marker="x")
    plt.xlabel("Normal force FN (N)")
    plt.ylabel("MRR (mm/min)")
    plt.title("Experimental vs ODE MRR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# -----------------------------
# Main execution
# -----------------------------


def main():
    df = create_experiment_dataframe()
    print("Experimental dataset:")
    print(df.to_string(index=False))

    a_MRR, b_MRR, r2_MRR = fit_mrr_regression(df)
    print(f"\nFitted linear MRR(FN): a = {a_MRR:.4f}, b = {b_MRR:.4f}, R^2 = {r2_MRR:.4f}")
    validate_mrr_regression(df, a_MRR, b_MRR)

    p = get_default_parameters()
    v_rel_fit = 15.0
    hardness_fit = 0.7
    grit_fit = 180.0
    p, diag = fit_material_removal_parameters(df, p, v_rel_fit, hardness_fit, grit_fit)
    print("\nFitted material removal parameters:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")
    validate_material_removal_law(df, p, v_rel_fit, hardness_fit, grit_fit)

    plot_mrr_vs_FN(df, a_MRR, b_MRR, "mrr_vs_FN.png")
    plot_validation_mrr(df, p, v_rel_fit, hardness_fit, grit_fit, "mrr_validation.png")

    FN_cases = [18.78615, 21.69972]
    t_sim = 30.0
    df_edge = simulate_edge_scenario(FN_cases, t_sim, p, theta_deg=90.0, scenario_name="right_angle")
    df_bevel = simulate_edge_scenario(FN_cases, t_sim, p, theta_deg=45.0, scenario_name="bevel_edge")

    print("\nSimulation comparison (right angle vs bevel):")
    comparison = pd.concat([df_edge.assign(edge_type="90deg"), df_bevel.assign(edge_type="45deg")], ignore_index=True)
    print(comparison)

    all_params = p.__dict__
    print("\nFinal model parameters:")
    for key, val in all_params.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
