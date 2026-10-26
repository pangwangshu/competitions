"""Sanity-check downloaded sample DICOMs: metadata + montage figures.

Walks data/sample/train_series/{study}/{series}/*.dcm, prints one line per
series (shape, spacing, orientation hints from ImageOrientationPatient) and
writes one montage PNG per study to reports/figures/.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsna_knee.constants import REPORTS_DIR, SAMPLE_DIR

FIG_DIR = REPORTS_DIR / "figures"

# ImageOrientationPatient cosine -> anatomical plane of the slice stack.
def plane_from_iop(iop) -> str:
    if iop is None:
        return "?"
    row, col = np.array(iop[:3]), np.array(iop[3:6])
    normal = np.cross(row, col)
    axis = int(np.argmax(np.abs(normal)))
    return ["Sagittal", "Coronal", "Axial"][axis]


def main() -> None:
    root = SAMPLE_DIR / "train_series"
    if not root.exists():
        raise SystemExit(f"no sample data at {root}; run scripts/download_sample.py first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        study_slices: dict[str, list[tuple[float, object]]] = {}
        for series_dir in sorted(study_dir.iterdir()):
            dcm_files = sorted(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            ds0 = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            insts = []
            for fp in dcm_files:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                insts.append((float(getattr(ds, "InstanceNumber", 0) or 0), fp))
                n_files += 1
            insts.sort()
            shape = f"{getattr(ds0, 'Rows', '?')}x{getattr(ds0, 'Columns', '?')}"
            plane = plane_from_iop(getattr(ds0, "ImageOrientationPatient", None))
            study_slices[series_dir.name[-8:]] = [
                (i, fp) for i, fp in insts
            ]
            print(
                f"{study_dir.name[-8:]} series {series_dir.name[-8:]}: "
                f"{len(dcm_files)} slices {shape} {plane:<8} "
                f"desc={getattr(ds0, 'SeriesDescription', '-')!r}"
            )

        # montage: middle slice of every series, up to 12 per row
        mids = []
        for name, insts in study_slices.items():
            fp = insts[len(insts) // 2][1]
            ds = pydicom.dcmread(fp)
            mids.append((name, ds.pixel_array.astype(np.float32)))
        if not mids:
            continue
        cols = min(len(mids), 6)
        rows = int(np.ceil(len(mids) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
        for ax, (name, img) in zip(axes.flat, mids):
            ax.imshow(img, cmap="gray")
            ax.set_title(name, fontsize=6)
            ax.axis("off")
        for ax in axes.flat[len(mids):]:
            ax.axis("off")
        fig.suptitle(study_dir.name[-8:], fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"study_{study_dir.name[-8:]}.png", dpi=120)
        plt.close(fig)

    print(f"\nread {n_files} DICOMs OK; montages in {FIG_DIR}")


if __name__ == "__main__":
    main()
