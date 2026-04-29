from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PSAParams:
    # Bed geometry
    L: float = 1.0
    N: int = 80
    eps: float = 0.38
    rho_s: float = 900.0

    # Operating condition
    u: float = 0.08
    T: float = 298.15
    P_ads: float = 5.0
    P_des: float = 0.2

    # Feed / purge composition
    y_CO2_feed: float = 0.15
    y_CO2_purge: float = 0.0

    # Langmuir isotherm parameters
    qmax: float = 4.0
    b: float = 0.8

    # LDF kinetics
    k_LDF: float = 0.08

    # Numerical
    dt: float = 0.05


DEFAULT_PARAMS = PSAParams()


def langmuir_qstar(y_co2: np.ndarray, pressure_bar: float, qmax: float, b: float) -> np.ndarray:
    """Return equilibrium CO2 loading [mol/kg] from a Langmuir isotherm."""
    p_co2 = y_co2 * pressure_bar
    return qmax * b * p_co2 / (1.0 + b * p_co2)


def bed_step(
    y: np.ndarray,
    q: np.ndarray,
    y_in: float,
    pressure_bar: float,
    params: PSAParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a simplified 1D packed-bed PSA model by one explicit time step."""
    dz = params.L / params.N

    y_new = y.copy()
    q_new = q.copy()

    q_star = langmuir_qstar(y, pressure_bar, params.qmax, params.b)
    dqdt = params.k_LDF * (q_star - q)

    for i in range(params.N):
        y_up = y_in if i == 0 else y[i - 1]
        convection = -params.u * (y[i] - y_up) / dz

        # The sink uses a small scaling factor to keep the simplified model stable.
        sink = -((1.0 - params.eps) / params.eps) * params.rho_s * dqdt[i] * 1e-3

        y_new[i] = y[i] + params.dt * (convection + sink)
        q_new[i] = q[i] + params.dt * dqdt[i]

    y_new = np.clip(y_new, 0.0, 1.0)
    q_new = np.clip(q_new, 0.0, params.qmax)
    return y_new, q_new


def run_step(
    y: np.ndarray,
    q: np.ndarray,
    y_in: float,
    pressure_bar: float,
    duration_s: float,
    params: PSAParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one PSA sub-step for a specified duration."""
    n_steps = max(1, int(duration_s / params.dt))
    outlet_history = np.empty(n_steps)

    for step in range(n_steps):
        y, q = bed_step(y, q, y_in, pressure_bar, params)
        outlet_history[step] = y[-1]

    return y, q, outlet_history


def run_psa_cycle(
    t_ads: float = 80.0,
    t_des: float = 80.0,
    t_purge: float = 40.0,
    t_repress: float = 20.0,
    n_cycles: int = 10,
    params: PSAParams = DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run repeated PSA cycles and return cycle-level KPIs."""
    y = np.zeros(params.N)
    q = np.zeros(params.N)
    records: list[dict[str, float]] = []

    for cycle in range(1, n_cycles + 1):
        y, q, outlet_ads = run_step(y, q, params.y_CO2_feed, params.P_ads, t_ads, params)
        y, q, outlet_dep = run_step(
            y,
            q,
            y_in=0.0,
            pressure_bar=(params.P_ads + params.P_des) / 2.0,
            duration_s=t_repress,
            params=params,
        )
        y, q, outlet_des = run_step(y, q, params.y_CO2_purge, params.P_des, t_des + t_purge, params)
        y, q, outlet_rep = run_step(y, q, params.y_CO2_feed, params.P_ads, t_repress, params)

        co2_product = np.trapezoid(outlet_des, dx=params.dt)
        co2_loss = np.trapezoid(outlet_ads, dx=params.dt)
        total_cycle_time = t_ads + t_des + t_purge + t_repress

        records.append(
            {
                "cycle": cycle,
                "purity": float(np.mean(outlet_des)),
                "recovery": float(co2_product / (co2_product + co2_loss + 1e-9)),
                "productivity": float(co2_product / total_cycle_time),
                "energy_proxy": float(params.P_ads / max(params.P_des, 1e-6) + 0.01 * t_des + 0.005 * t_purge),
                "q_avg": float(np.mean(q)),
                "q_final_avg": float(np.mean(q)),
                "outlet_ads_final": float(outlet_ads[-1]),
                "outlet_dep_final": float(outlet_dep[-1]),
                "outlet_des_avg": float(np.mean(outlet_des)),
                "outlet_rep_final": float(outlet_rep[-1]),
            }
        )

    return pd.DataFrame(records), y, q


def save_cycle_plots(
    df_result: pd.DataFrame,
    y_final: np.ndarray,
    q_final: np.ndarray,
    params: PSAParams,
    output_dir: Path,
) -> None:
    """Save KPI and final bed profile plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    z = np.linspace(0.0, params.L, params.N)

    plt.figure(figsize=(8, 4.5))
    plt.plot(df_result["cycle"], df_result["purity"], marker="o", label="Purity")
    plt.plot(df_result["cycle"], df_result["recovery"], marker="s", label="Recovery")
    plt.xlabel("Cycle")
    plt.ylabel("KPI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cycle_kpi.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(z, y_final, label="Gas-phase CO2 mole fraction")
    plt.xlabel("Bed position [m]")
    plt.ylabel("y_CO2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bed_profile_y.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(z, q_final, label="Solid loading")
    plt.xlabel("Bed position [m]")
    plt.ylabel("q_CO2 [mol/kg]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bed_profile_q.png", dpi=150)
    plt.close()
