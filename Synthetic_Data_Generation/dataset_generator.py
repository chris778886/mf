from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import qmc

from psa_simulator import DEFAULT_PARAMS, PSAParams, run_psa_cycle


def _print_generation_progress(current: int, total: int) -> None:
    if current % 10 == 0 or current == total:
        print(f"{current}회 생성완료 ({current}/{total})")


def generate_synthetic_dataset(
    n_samples: int = 100,
    n_cycles: int = 10,
    t_repress: float = 20.0,
    base_params: PSAParams = DEFAULT_PARAMS,
    seed: int = 100,
) -> pd.DataFrame:
    """Generate an LHS-based synthetic PSA dataset."""
    variables: dict[str, tuple[float, float]] = {
        "P_ads": (2.0, 8.0),
        "P_des": (0.05, 1.0),
        "t_ads": (30.0, 200.0),
        "t_des": (30.0, 200.0),
        "t_purge": (10.0, 100.0),
        "u": (0.02, 0.15),
        "qmax": (2.0, 6.0),
        "b": (0.2, 1.5),
        "k_LDF": (0.02, 0.2),
        "L": (0.5, 2.0),
        "eps": (0.30, 0.50),
    }

    sampler = qmc.LatinHypercube(d=len(variables), seed=seed)
    sample = sampler.random(n=n_samples)

    lower = [bounds[0] for bounds in variables.values()]
    upper = [bounds[1] for bounds in variables.values()]
    sample_scaled = qmc.scale(sample, lower, upper)

    dataset: list[dict[str, float]] = []
    variable_names = list(variables.keys())

    for sample_id, row in enumerate(sample_scaled, start=1):
        updates = {name: float(value) for name, value in zip(variable_names, row)}
        case_params = replace(
            base_params,
            P_ads=updates["P_ads"],
            P_des=updates["P_des"],
            u=updates["u"],
            qmax=updates["qmax"],
            b=updates["b"],
            k_LDF=updates["k_LDF"],
            L=updates["L"],
            eps=updates["eps"],
        )

        result, _, q_final = run_psa_cycle(
            t_ads=updates["t_ads"],
            t_des=updates["t_des"],
            t_purge=updates["t_purge"],
            t_repress=t_repress,
            n_cycles=n_cycles,
            params=case_params,
        )
        final_cycle = result.iloc[-1].to_dict()

        dataset.append(
            {
                **updates,
                "t_repress": t_repress,
                **final_cycle,
                "final_loading_mean": float(np.mean(q_final)),
                "final_loading_max": float(np.max(q_final)),
            }
        )
        _print_generation_progress(sample_id, n_samples)

    return pd.DataFrame(dataset)


def generate_synthetic_dataset_with_logs(
    n_samples: int = 100,
    n_cycles: int = 10,
    t_repress: float = 20.0,
    base_params: PSAParams = DEFAULT_PARAMS,
    seed: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full synthetic dataset and per-cycle logs for every sample."""
    variables: dict[str, tuple[float, float]] = {
        "P_ads": (2.0, 8.0),
        "P_des": (0.05, 1.0),
        "t_ads": (30.0, 200.0),
        "t_des": (30.0, 200.0),
        "t_purge": (10.0, 100.0),
        "u": (0.02, 0.15),
        "qmax": (2.0, 6.0),
        "b": (0.2, 1.5),
        "k_LDF": (0.02, 0.2),
        "L": (0.5, 2.0),
        "eps": (0.30, 0.50),
    }

    sampler = qmc.LatinHypercube(d=len(variables), seed=seed)
    sample = sampler.random(n=n_samples)

    lower = [bounds[0] for bounds in variables.values()]
    upper = [bounds[1] for bounds in variables.values()]
    sample_scaled = qmc.scale(sample, lower, upper)

    dataset: list[dict[str, float]] = []
    cycle_logs: list[pd.DataFrame] = []
    variable_names = list(variables.keys())

    for sample_id, row in enumerate(sample_scaled, start=1):
        updates = {name: float(value) for name, value in zip(variable_names, row)}
        case_params = replace(
            base_params,
            P_ads=updates["P_ads"],
            P_des=updates["P_des"],
            u=updates["u"],
            qmax=updates["qmax"],
            b=updates["b"],
            k_LDF=updates["k_LDF"],
            L=updates["L"],
            eps=updates["eps"],
        )

        result, _, q_final = run_psa_cycle(
            t_ads=updates["t_ads"],
            t_des=updates["t_des"],
            t_purge=updates["t_purge"],
            t_repress=t_repress,
            n_cycles=n_cycles,
            params=case_params,
        )
        final_cycle = result.iloc[-1].to_dict()

        dataset.append(
            {
                "sample_id": sample_id,
                **updates,
                "t_repress": t_repress,
                **final_cycle,
                "final_loading_mean": float(np.mean(q_final)),
                "final_loading_max": float(np.max(q_final)),
            }
        )

        result_with_context = result.copy()
        result_with_context.insert(0, "sample_id", sample_id)
        for name, value in updates.items():
            result_with_context[name] = value
        result_with_context["t_repress"] = t_repress
        cycle_logs.append(result_with_context)
        _print_generation_progress(sample_id, n_samples)

    return pd.DataFrame(dataset), pd.concat(cycle_logs, ignore_index=True)
