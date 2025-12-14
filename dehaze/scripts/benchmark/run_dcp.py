import sys
import csv
from pathlib import Path

import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

# ============================================================
# Add project root to Python path to allow importing src.*
# scripts/benchmark/run_dcp_ihaze.py -> project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Use your own DCP implementation
from src.algorithms.dcp import dehaze


# ======================== Configuration (I-HAZE) ========================

# Dataset paths
HAZY_DIR = Path("data/raw/ihaze/hazy")
GT_DIR   = Path("data/raw/ihaze/gt")

# Output directory
OUTPUT_DIR = Path("data/results/dcp_full_ihaze")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics CSV
METRICS_CSV = OUTPUT_DIR / "metrics_dcp_ihaze.csv"

# Resize long side (keep consistent with CLAHE / RIDCP)
LONG_SIDE = 720   # can be changed to 512 / 1024

# DCP parameters (keep consistent with your experiments)
DCP_PATCH_SIZE = 15
DCP_OMEGA = 0.95
DCP_T0 = 0.1
DCP_USE_GUIDED = True
DCP_GUIDED_RADIUS = 40
DCP_GUIDED_EPS = 0.001


# ======================== Utility Functions ========================

def resize_long_side(img, long_side):
    """
    Resize image so that the longer side equals long_side,
    while keeping the aspect ratio.
    """
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img

    scale = long_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)


def compute_ssim(pred, gt):
    return structural_similarity(gt, pred, channel_axis=-1, data_range=255)


def compute_deltaE00(pred, gt):
    pred_lab = color.rgb2lab(pred / 255.0)
    gt_lab   = color.rgb2lab(gt   / 255.0)
    delta = color.deltaE_ciede2000(gt_lab, pred_lab)
    return float(delta.mean())


# ======================== Main Pipeline ========================

def main():
    hazy_files = sorted(HAZY_DIR.glob("*.jpg"))

    if not hazy_files:
        print(f"[I-HAZE] ERROR: No hazy images found in {HAZY_DIR}")
        return

    print(f"[I-HAZE] Found {len(hazy_files)} hazy images. Start processing...")
    rows = []

    for hazy_path in hazy_files:
        fname = hazy_path.name

        # Example: 01_indoor_hazy.jpg -> 01_indoor_GT.jpg
        gt_name = fname.replace("_hazy", "_GT")
        gt_path = GT_DIR / gt_name

        if not gt_path.exists():
            print(f"[I-HAZE] WARNING: GT not found: {gt_path}, skip")
            continue

        hazy = cv2.imread(str(hazy_path), cv2.IMREAD_COLOR)
        gt   = cv2.imread(str(gt_path),   cv2.IMREAD_COLOR)

        if hazy is None or gt is None:
            print(f"[I-HAZE] WARNING: Failed to read {fname}, skip")
            continue

        print(f"\n[I-HAZE] Processing {fname}")
        print(f"  Original size: hazy={hazy.shape}, gt={gt.shape}")

        # Resize
        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small   = resize_long_side(gt,   LONG_SIDE)
        print(f"  Resized size : hazy={hazy_small.shape}, gt={gt_small.shape}")

        # ---------------- DCP dehazing ----------------
        out = dehaze(
            hazy_small,
            patch_size=DCP_PATCH_SIZE,
            omega=DCP_OMEGA,
            t0=DCP_T0,
            use_guided_filter=DCP_USE_GUIDED,
            guided_radius=DCP_GUIDED_RADIUS,
            guided_eps=DCP_GUIDED_EPS,
        )

        # Save result
        save_path = OUTPUT_DIR / fname
        cv2.imwrite(str(save_path), out)
        print(f"  Saved result to: {save_path}")

        # Compute metrics in RGB space
        out_rgb = cv2.cvtColor(out,      cv2.COLOR_BGR2RGB)
        gt_rgb  = cv2.cvtColor(gt_small, cv2.COLOR_BGR2RGB)

        psnr_val = compute_psnr(out_rgb, gt_rgb)
        ssim_val = compute_ssim(out_rgb, gt_rgb)
        de_val   = compute_deltaE00(out_rgb, gt_rgb)

        print(f"  PSNR    = {psnr_val:.2f}")
        print(f"  SSIM    = {ssim_val:.4f}")
        print(f"  DeltaE  = {de_val:.2f}")

        rows.append([fname, psnr_val, ssim_val, de_val])

    # Write CSV
    if rows:
        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)

        print(f"\n[I-HAZE] Metrics saved to: {METRICS_CSV}")
    else:
        print("\n[I-HAZE] No images were successfully processed.")


if __name__ == "__main__":
    main()
    print("\n[DCP + I-HAZE] Finished successfully.")
