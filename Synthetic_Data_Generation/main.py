import argparse
from pathlib import Path

from dataset_generator import generate_synthetic_dataset_with_logs
from psa_simulator import DEFAULT_PARAMS, run_psa_cycle, save_cycle_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified PSA packed-bed simulator and synthetic dataset generator.")
    parser.add_argument("--samples", type=int, default=200, help="Number of LHS samples for synthetic dataset generation.")
    parser.add_argument("--cycles", type=int, default=15, help="Number of PSA cycles for the example simulation.")
    parser.add_argument("--dataset-cycles", type=int, default=10, help="Number of cycles per synthetic dataset sample.")
    parser.add_argument("--plots", action="store_true", help="Save KPI and final bed profile plots.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for CSV and plot outputs.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for LHS sampling.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dataset, df_cycle_logs = generate_synthetic_dataset_with_logs(
        n_samples=args.samples,
        n_cycles=args.dataset_cycles,
        t_repress=20.0,
        base_params=DEFAULT_PARAMS,
        seed=args.seed,
    )
    df_cycle_logs.to_csv(args.output_dir / "psa_cycle_results.csv", index=False)
    df_dataset.to_csv(args.output_dir / "psa_synthetic_dataset.csv", index=False)

    df_result, y_final, q_final = run_psa_cycle(
        t_ads=80.0,
        t_des=80.0,
        t_purge=40.0,
        t_repress=20.0,
        n_cycles=args.cycles,
        params=DEFAULT_PARAMS,
    )

    if args.plots:
        save_cycle_plots(df_result, y_final, q_final, DEFAULT_PARAMS, args.output_dir)

    print("Cycle results saved to:", args.output_dir / "psa_cycle_results.csv")
    print("Synthetic dataset saved to:", args.output_dir / "psa_synthetic_dataset.csv")
    print()
    print("Final cycle KPIs:")
    print(df_cycle_logs.tail(3).to_string(index=False))
    print()
    print("Synthetic dataset preview:")
    print(df_dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
