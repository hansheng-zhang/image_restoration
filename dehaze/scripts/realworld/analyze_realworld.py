import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Input CSV produced by scripts/realworld/run_realworld.py
CSV_PATH = Path("data/results/real_haze/real_haze_brisque.csv")

# Output files
OUT_DIR = Path("data/results/real_haze")
OUT_FIG = OUT_DIR / "real_haze_brisque_bar.png"
OUT_SUMMARY_CSV = OUT_DIR / "real_haze_brisque_mean.csv"


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    cols = ["hazy_brisque", "clahe_brisque", "dcp_brisque", "ridcp_brisque"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in CSV: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    means = df[cols].mean(numeric_only=True)

    print("Mean BRISQUE (lower is better):")
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    # Save mean summary CSV (useful for report/ppt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = means.reset_index()
    summary_df.columns = ["metric", "mean_brisque"]
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\n[INFO] Mean summary saved to: {OUT_SUMMARY_CSV}")

    # Plot bar chart
    methods = ["Hazy", "CLAHE", "DCP", "RIDCP"]
    values = [
        means["hazy_brisque"],
        means["clahe_brisque"],
        means["dcp_brisque"],
        means["ridcp_brisque"],
    ]

    plt.figure()
    plt.bar(methods, values)
    plt.ylabel("Mean BRISQUE (lower is better)")
    plt.title("Real-World Hazy Images: BRISQUE Comparison")
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print(f"[INFO] Bar plot saved to: {OUT_FIG}")


if __name__ == "__main__":
    main()
