"""Run the private CPU cache campaign, resume receipts, and archive only metadata."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kx import ROOT, STATE, expected_kernel, get_api, push_kernel, pull_kernel, read_json, wait_kernel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--interval", type=float, default=90)
    args = parser.parse_args()
    if args.parallel < 1 or args.interval < 1:
        parser.error("parallel and interval must be positive")
    experiment = ROOT / "experiments" / "S1"
    selection = read_json(experiment / "cache_plan.json")["selection"]
    publication = read_json(STATE / "code-publication.json")
    api = get_api()
    pending, active, complete = [], {}, {}
    started = time.monotonic()

    def receipt_path(shard):
        return STATE / "kernels" / f"rsna-knee-cache-shard-{shard}.json"

    def directory(shard):
        path = experiment / "shards" / str(shard)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshot():
        write_json(experiment / "campaign.json", {"pending": pending, "active": active,
                   "complete": complete, "elapsed_seconds": time.monotonic() - started,
                   "code_dataset_version": publication["version"], "parallel": args.parallel})

    def archive_summary(shard, version):
        report = read_json(directory(shard) / f"shard{shard}.json")
        return {"kernel": f"wangshu/rsna-knee-cache-shard-{shard}", "version": version,
                "series_count": report["series_count"], "bytes": report["output_bytes"],
                "failed": report["failed_count"], "wall_seconds": report["wall_seconds"]}

    for shard in range(selection["shards"]):
        if receipt_path(shard).exists():
            receipt = read_json(receipt_path(shard))
            config = read_json(ROOT / "kaggle" / f"cache-shard-{shard}" / "config.json")
            if receipt["code_archive_sha256"] != publication["manifest"]["archive_sha256"] or receipt["config"] != config:
                raise ValueError(f"shard {shard} has a different source/config receipt; inspect before rerunning")
            output = directory(shard) / "kx-output.json"
            report = directory(shard) / f"shard{shard}.json"
            if output.exists() and report.exists() and read_json(output)["version"] == receipt["version"]:
                expected_kernel(api, f"{receipt['kernel']}/{receipt['version']}")
                complete[shard] = archive_summary(shard, receipt["version"])
            else:
                active[shard] = receipt["version"]
        else:
            pending.append(shard)

    while pending or active:
        ready = []
        for shard, version in list(active.items()):
            slug = f"wangshu/rsna-knee-cache-shard-{shard}"
            status = wait_kernel(api, f"{slug}/{version}", once=True)
            shutil.copyfile(receipt_path(shard), directory(shard) / "kernel-publication.json")
            write_json(directory(shard) / "status.json", status)
            if status["status"] == "COMPLETE":
                ready.append((shard, version))
                active.pop(shard)
        while pending and len(active) < args.parallel:
            shard = pending[0]
            kernel = ROOT / "kaggle" / f"cache-shard-{shard}"
            # Mutating calls are deliberately sequential; source staging is shared.
            receipt = push_kernel(api, kernel, read_json(kernel / "config.json"))
            active[shard] = receipt["version"]
            pending.pop(0)
            write_json(directory(shard) / "kernel-publication.json", receipt)
            print("LAUNCHED " + json.dumps(receipt), flush=True)
            snapshot()
        for shard, version in ready:
            slug = f"wangshu/rsna-knee-cache-shard-{shard}"
            pull_kernel(api, f"{slug}/{version}", directory(shard), file_pattern=r"\.(csv|json)$")
            complete[shard] = archive_summary(shard, version)
            print("ARCHIVED " + json.dumps(complete[shard]), flush=True)
            snapshot()
        snapshot()
        print(json.dumps({"complete": len(complete), "active": list(active), "pending": pending}), flush=True)
        if pending or active:
            wake = time.monotonic() + args.interval
            while time.monotonic() < wake:
                time.sleep(min(30, max(0, wake - time.monotonic())))
    print("S1 CACHE CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
