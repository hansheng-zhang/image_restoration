import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import csv
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color

from src.algorithms.clahe import dehaze


# ===================== Config =====================

LONG_SIDE = 720

CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_CLIP_LIMIT = 2.0

DATASETS = {
    "ihaze": {
        "hazy_dir": Path("data/raw/ihaze/hazy"),
        "gt_dir":   Path("data/raw/ihaze/gt"),
        "out_dir":  Path("data/results/clahe_full_ihaze"),
        "csv":      "metrics_clahe_ihaze.csv",
        "prefix":   "[I-HAZE]",
    },
    "ohaze": {
        "hazy_dir": Path("data/raw/ohaze/hazy"),
        "gt_dir":   Path("data/raw/ohaze/gt"),
        "out_dir":  Path("data/results/clahe_full_ohaze"),
        "csv":      "metrics_clahe_ohaze.csv",
        "prefix":   "[O-HAZE]",
    },
}


# ===================== Utils =====================

def resize_long_side(img, long_side):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=255)


def ssim(pred, gt):
    return structural_similarity(gt, pred, channel_axis=-1, data_range=255)


def deltaE00(pred, gt):
    pred_lab = color.rgb2lab(pred / 255.0)
    gt_lab   = color.rgb2lab(gt   / 255.0)
    return float(color.deltaE_ciede2000(gt_lab, pred_lab).mean())


def hazy_to_gt(name):
    return name.replace("_hazy", "_GT")


# ===================== Main =====================

def run_one_dataset(cfg):
    hazy_dir = cfg["hazy_dir"]
    gt_dir = cfg["gt_dir"]
    out_dir = cfg["out_dir"]
    prefix = cfg["prefix"]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / cfg["csv"]

    hazy_files = sorted(hazy_dir.glob("*.jpg"))
    if not hazy_files:
        print(f"{prefix} No hazy images found in {hazy_dir}")
        return

    rows = []
    print(f"{prefix} Found {len(hazy_files)} images")

    for hazy_path in hazy_files:
        fname = hazy_path.name
        gt_path = gt_dir / hazy_to_gt(fname)

        if not gt_path.exists():
            print(f"{prefix} GT not found: {gt_path}, skip")
            continue

        hazy = cv2.imread(str(hazy_path))
        gt = cv2.imread(str(gt_path))

        if hazy is None or gt is None:
            print(f"{prefix} Failed to read {fname}, skip")
            continue

        hazy_small = resize_long_side(hazy, LONG_SIDE)
        gt_small = resize_long_side(gt, LONG_SIDE)

        out = dehaze(
            hazy_small,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
            clip_limit=CLAHE_CLIP_LIMIT,
        )

        cv2.imwrite(str(out_dir / fname), out)

        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_small, cv2.COLOR_BGR2RGB)

        p = psnr(out_rgb, gt_rgb)
        s = ssim(out_rgb, gt_rgb)
        d = deltaE00(out_rgb, gt_rgb)

        print(f"{prefix} {fname}: PSNR={p:.2f}, SSIM={s:.4f}, ΔE={d:.2f}")
        rows.append([fname, p, s, d])

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "PSNR", "SSIM", "DeltaE00"])
            writer.writerows(rows)
        print(f"{prefix} Metrics saved to {csv_path}")


def main():
    for cfg in DATASETS.values():
        run_one_dataset(cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
