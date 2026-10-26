"""Download a sample of training studies (all their DICOM series) from Kaggle.

The competition publishes data as one file per DICOM, so sampling means:
pick studies -> look up their file paths in the cached file listing
(data/filelist_pages, produced by scripts/list_all_files.sh) -> download each
file in parallel with `kaggle competitions download -f`.

Usage:
    uv run python scripts/download_sample.py --n-studies 10 --workers 8
    uv run python scripts/download_sample.py --n-studies 5 --from-filelist   # only studies already in the cached listing
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsna_knee.constants import COMPETITION, FILELIST_DIR, LABELS, RAW_DIR, SAMPLE_DIR
from rsna_knee.metadata import load_train, load_train_series


def collect_listed_files(filelist_dir: Path) -> dict[str, list[str]]:
    """Map StudyInstanceUID -> listed train_series DICOM paths, from cached pages."""
    study_to_files: dict[str, list[str]] = defaultdict(list)
    pages = sorted(filelist_dir.glob("page_*.csv"))
    if not pages:
        raise SystemExit(
            f"No cached listing pages in {filelist_dir}. Run scripts/list_all_files.sh first."
        )
    for page in pages:
        with open(page) as f:
            lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#token:")]
        for ln in lines:
            first = ln.split(",", 1)[0].strip('"')
            if not first.startswith("train_series/"):
                continue
            parts = first.split("/")  # train_series/{study}/{series}/{sop}.dcm
            if len(parts) == 4:
                study_to_files[parts[1]].append(first)
    return study_to_files


class AuthLost(Exception):
    """Raised when the Kaggle token stops working — abort instead of hammering 401s."""


def download_file(remote_path: str, out_root: Path) -> tuple[str, bool]:
    parts = remote_path.split("/")  # train_series/{study}/{series}/{sop}.dcm
    dest_dir = out_root.joinpath(*parts[:3])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / parts[3]
    if dest.exists() and dest.stat().st_size > 0:
        return remote_path, True
    cmd = [
        "kaggle", "competitions", "download", COMPETITION,
        "-f", remote_path, "-p", str(dest_dir), "--force",
    ]
    proc = None
    for attempt in range(2):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return remote_path, True
        if "Authentication required" in proc.stderr:
            raise AuthLost("kaggle token expired/revoked — run: kaggle auth login")
    print(f"FAILED: {remote_path}\n{proc.stderr[-300:]}", file=sys.stderr)
    return remote_path, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-studies", type=int, default=10,
                    help="number of UNLABELED studies to sample (labeled ones are additive)")
    ap.add_argument("--include-labeled", action="store_true", default=True,
                    help="always include all gold-labeled studies (only 58 exist; ~7 GB)")
    ap.add_argument("--no-labeled", dest="include_labeled", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=SAMPLE_DIR)
    ap.add_argument("--filelist-dir", type=Path, default=FILELIST_DIR)
    ap.add_argument(
        "--from-filelist",
        action="store_true",
        help="sample only among studies already present in the cached listing "
        "(useful while the full listing is still downloading)",
    )
    args = ap.parse_args()

    train = load_train(RAW_DIR)
    series = load_train_series(RAW_DIR)
    listed = collect_listed_files(args.filelist_dir)

    labeled_ids = train.loc[train["labeled"], "StudyInstanceUID"]
    unlabeled_ids = train.loc[~train["labeled"], "StudyInstanceUID"]
    if args.from_filelist:
        labeled_ids = labeled_ids[labeled_ids.isin(listed)]
        unlabeled_ids = unlabeled_ids[unlabeled_ids.isin(listed)]

    picked = []
    if args.include_labeled:
        picked.extend(sorted(labeled_ids))
        print(f"including all {len(labeled_ids)} gold-labeled studies "
              f"({labeled_ids.shape[0]}/{train['labeled'].sum()} available in listing)")
    n_available = unlabeled_ids.nunique()
    if args.n_studies > 0:
        picked.extend(sorted(unlabeled_ids.drop_duplicates().sample(
            n=min(args.n_studies, n_available), random_state=args.seed
        )))
    picked = sorted(set(picked))
    print(f"total: {len(picked)} studies ({n_available} unlabeled available to sample from)")

    files: list[str] = []
    for study in picked:
        files.extend(sorted(listed[study]))
    print(f"{len(files)} DICOM files to download "
          f"(~{len(files) * 0.6 / 1024:.1f} GB at ~0.6 MB/file)")

    ok = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(download_file, fp, args.out) for fp in files]
            for i, fut in enumerate(cf.as_completed(futures), 1):
                _, success = fut.result()
                ok += success
                if i % 200 == 0 or i == len(files):
                    print(f"  {i}/{len(files)} done ({ok} ok)")
    except AuthLost as e:
        print(f"\nABORTED after {ok} ok: {e}", file=sys.stderr)
        print("Re-auth, then rerun this command — existing files are skipped.")
        sys.exit(2)

    print(f"downloaded {ok}/{len(files)} files")

    label_rows = train.set_index("StudyInstanceUID").loc[picked].reset_index()[
        ["StudyInstanceUID", "labeled", *LABELS, "n_positives"]
    ]
    label_rows[LABELS] = label_rows[LABELS].astype(object).where(
        label_rows[LABELS].notna(), None
    )
    manifest = {
        "seed": args.seed,
        "n_studies": len(picked),
        "studies": picked,
        "label_rows": label_rows.to_dict(orient="records"),
        "series_rows": series[series["StudyInstanceUID"].isin(picked)].to_dict(
            orient="records"
        ),
        "files": files,
        "downloaded_ok": ok,
    }
    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
