"""Download label sources at the versions and SHA-256 hashes recorded in S1."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kx import ROOT

MANIFEST = ROOT / "experiments" / "S1" / "source_datasets.json"


def fetch(api, manifest=MANIFEST, destination=None):
    destination = Path(destination or ROOT / "data" / "ext")
    for row in json.loads(Path(manifest).read_text()):
        owner, slug = row["dataset"].split("/")
        folder = destination / f"{owner}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / row["filename"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            api.dataset_download_file(f"{row['dataset']}/{row['version']}", row["filename"],
                                      path=str(folder), force=True, quiet=True)
        if not path.is_file():
            raise FileNotFoundError(f"Expected downloaded CSV at {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}; refusing a changed label source")
        print(f"Verified {row['dataset']}/{row['version']}: {row['filename']}", flush=True)


def main():
    from kaggle import api  # any authenticated account with access to the public sources
    fetch(api)


if __name__ == "__main__":
    main()
