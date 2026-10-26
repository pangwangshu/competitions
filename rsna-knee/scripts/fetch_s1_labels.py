"""Download the four specified public label CSVs, pinned to observed versions."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kx import ROOT, get_api, write_json
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

SOURCES = [
    ("stevenleehans", "rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv"),
    ("pilkwang", "rsna-knee-llm-labels", "report_labels_v2.csv"),
    ("lixin73", "rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv"),
    ("laymond", "rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv"),
]


def main():
    api = get_api()
    rows = []
    for owner, slug, filename in SOURCES:
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = owner, slug
        with api.build_kaggle_client() as client:
            info = client.datasets.dataset_api_client.get_dataset(request)
        version = info.current_version_number
        destination = ROOT / "data" / "ext" / f"{owner}_{slug}"
        destination.mkdir(parents=True, exist_ok=True)
        api.dataset_download_file(f"{owner}/{slug}/{version}", filename, path=str(destination), force=True, quiet=True)
        path = destination / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected downloaded CSV at {path}")
        row = {"dataset": f"{owner}/{slug}", "version": version, "filename": filename,
               "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row)
        print(row, flush=True)
    write_json(ROOT / "experiments" / "S1" / "source_datasets.json", rows)


if __name__ == "__main__":
    main()
