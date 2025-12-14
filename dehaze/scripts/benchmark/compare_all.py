from pathlib import Path
import pandas as pd


DATASETS = {
    "ihaze": {
        "clahe": "data/results/clahe_full_ihaze/metrics_clahe_ihaze.csv",
        "dcp":   "data/results/dcp_full_ihaze/metrics_dcp_ihaze.csv",
        "ridcp": "data/results/ridcp_full_ihaze/metrics_ridcp_ihaze.csv",
        "out_compare": "data/results/compare_ihaze.csv",
        "out_avg":     "data/results/compare_ihaze_avg.csv",
    },
    "ohaze": {
        "clahe": "data/results/clahe_full_ohaze/metrics_clahe_ohaze.csv",
        "dcp":   "data/results/dcp_full_ohaze/metrics_dcp_ohaze.csv",
        "ridcp": "data/results/ridcp_full_ohaze/metrics_ridcp_ohaze.csv",
        "out_compare": "data/results/compare_ohaze.csv",
        "out_avg":     "data/results/compare_ohaze_avg.csv",
    },
}

SUMMARY_OUT = Path("data/results/compare_all_summary.csv")


def rename_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(columns={
        "PSNR": f"{prefix}_PSNR",
        "SSIM": f"{prefix}_SSIM",
        "DeltaE00": f"{prefix}_DE",
    })


def ensure_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def combine_one_dataset(name: str, paths: dict) -> pd.Series:
    # Make sure output folders exist
    ensure_parent(paths["out_compare"])
    ensure_parent(paths["out_avg"])
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Read and rename
    df_c = rename_cols(pd.read_csv(paths["clahe"]), "CLAHE")
    df_d = rename_cols(pd.read_csv(paths["dcp"]), "DCP")
    df_r = rename_cols(pd.read_csv(paths["ridcp"]), "RIDCP")

    # Merge on filename
    df = df_c.merge(df_d, on="filename").merge(df_r, on="filename")

    # Save per-image comparison table
    df.to_csv(paths["out_compare"], index=False)
    print(f"[{name.upper()}] Saved per-image comparison table -> {paths['out_compare']}")

    # Compute and save averages
    avg = df.mean(numeric_only=True)

    avg_df = avg.reset_index()
    avg_df.columns = ["metric", "value"]
    avg_df.to_csv(paths["out_avg"], index=False)
    print(f"[{name.upper()}] Saved average table -> {paths['out_avg']}")

    return avg


def main():
    summary = {}

    for name, paths in DATASETS.items():
        print(f"\n===== Processing {name.upper()} =====")
        avg = combine_one_dataset(name, paths)
        summary[name] = avg

    # Combine ihaze + ohaze into one summary table
    df_all = pd.DataFrame(summary)
    df_all.to_csv(SUMMARY_OUT, index=True)
    print(f"\nSaved overall summary table: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
