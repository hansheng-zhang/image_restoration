import sys
import csv
from pathlib import Path

import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# ============================================================
# Add project root to Python path to allow importing src.*
# scripts/benchmark/run_ridcp.py -> project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Your RIDCP wrapper
from src.algorithms.ridcp import dehaze


# ================== Global Config ==================

# Resize long side (keep consistent with CLAHE / DCP)
LONG_SIDE = 720  # can be 512 / 1024

# RIDCP repo root and weight path (relative to project root / CWD)
RIDCP_ROOT = "external/RIDCP_dehazing"
RIDCP_WEIGHT = "pretrained_models/pretrained_RIDCP.pth"
RIDCP_ALPHA = -21.25
RIDCP_USE_WEIGHT = True

# Dataset configs (input/output)
DATASETS = {
    "ihaze": {
        "hazy_dir": Path("data/raw/ihaze/hazy"),
        "gt_dir":   Path("data/raw/ihaze/gt"),
        "out_dir":  Path("data/results/ridcp_full_ihaze"),
        "csv":      Path("data/results/ridcp_full_ihaze/metrics_ridcp_ihaze.csv"),
        "prefix":   "[I-HAZE]",
    },
    "ohaze": {
        "hazy_dir": Path("data/raw/ohaze/hazy"),
        "gt_dir":   Path("data/raw/ohaze/gt"),
        "out_dir":  Path("data/results/ridcp_full_ohaze"),
        "csv":      Path("data/results/ridcp_full_ohaze/metrics_ridcp_ohaze.csv"),
        "prefix":   "[O-HAZE]",
    },
}


# ================== Utility Functions ==================

def resize_long_side(img, long_side):
    """Resize so that max(h, w) == long_side, keeping aspect ratio."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)


def ssim(pred, gt):
    return structural_similarity(gt, pred, channel_axis=-1, data_range=255)


def deltaE00(pred, gt):
    pred_lab = color.rgb2lab(pred / 255.0)
    gt_lab   = color.rgb2lab(gt   / 255.0)
    delta = color.deltaE_ciede2000(gt_lab, pred_lab)
    return float(delta.mean())


def hazy_to_gt_name(hazy_filename: str) -> str:
    """Assumption: xxx_hazy.jpg -> xxx_GT.jpg"""
    return hazy_filename.replace("_hazy", "_GT")


def make_save_name(fname: str) -> str:
    """Save as xxx_ridcp.jpg/png to avoid overwriting the original name."""
    if fname.lower().endswith(".jpg"):
        return fname[:-4] + "_ridcp.jpg"
    if fname.lower().endswith(".png"):
        return fname[:-4] + "_ridcp.png"
    return fname + "_ridcp"


# ================== Main Runner ==================

def run_one_dataset(name: str, hazy_dir: Path, gt_dir: Path, out_dir: Path, csv_path: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    hazy_files = sorted(hazy_dir.glob("*.jpg"))
    if not hazy_files:
        print(f"{prefix} ERROR: No hazy images found in: {hazy_dir}")
        return

    rows = []
    print(f"\n========== RIDCP on {name.upper()} ==========")
    print(f"{prefix} Found {len(hazy_files)} hazy images. Start processing...")

    for hazy_path in hazy_files:
        fname = hazy_path.name
        gt_name = hazy_to_gt_name(fname)
        gt_path = gt_dir / gt_name

        if not gt_path.exists():
            print(f"{prefix} WARNING: GT not found: {gt_path} -> skip")
            continue

        hazy = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        gt   = cv2.imread(str(gt_path),   cv2.IMREAD_COLOR)

        if hazy is None or gt is None:
            print(f"{prefix} WARNING: Failed to read: {hazy_path} or {gt_path} -> skip")
            continue

        print(f"\n{prefix} Processing {fname}")
        print(f"{prefix} Original size: hazy={hazy.shape}, gt={gt.shape}")

        # Resize hazy; resize GT to match hazy_small shape exactly
        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small = cv2.resize(
            gt,
            (hazy_small.shape[1], hazy_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        print(f"{prefix} Resized size : hazy={hazy_small.shape}, gt={gt_small.shape}")

        # Run RIDCP
        out = dehaze(
            hazy_small,
            ridcp_root=RIDCP_ROOT,
            weight_relpath=RIDCP_WEIGHT,
            alpha=RIDCP_ALPHA,
            use_weight=RIDCP_USE_WEIGHT,
            python_exec=sys.executable,
        )

        # Save output
        save_path = out_dir / make_save_name(fname)
        cv2.imwrite(str(save_path), out)
        print(f"{prefix} Saved result to: {save_path}")

        # Compute metrics (RGB)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        gt_rgb  = cv2.cvtColor(gt_small, cv2.COLOR_BGR2RGB)

        cur_psnr = psnr(out_rgb, gt_rgb)
        cur_ssim = ssim(out_rgb, gt_rgb)
        cur_de   = deltaE00(out_rgb, gt_rgb)

        print(f"{prefix} PSNR    = {cur_psnr:.2f}")
        print(f"{prefix} SSIM    = {cur_ssim:.4f}")
        print(f"{prefix} DeltaE  = {cur_de:.2f}")

        rows.append([fname, cur_psnr, cur_ssim, cur_de])

    # Write CSV
    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)
        print(f"\n{prefix} Metrics saved to: {csv_path}")
    else:
        print(f"\n{prefix} No samples processed successfully.")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_one_dataset(
            name=name,
            hazy_dir=cfg["hazy_dir"],
            gt_dir=cfg["gt_dir"],
            out_dir=cfg["out_dir"],
            csv_path=cfg["csv"],
            prefix=cfg["prefix"],
        )

    print("\nAll done. (RIDCP on I-HAZE + O-HAZE)")
