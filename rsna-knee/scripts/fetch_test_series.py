"""Extract only the members we need (default: test_series/) from the 265GB competition zip.

Uses HTTP range requests against the signed archive URL, so the whole zip is never
downloaded and only one authenticated Kaggle API call is made (per-file
`kaggle competitions download -f` burns one call per DICOM).

Usage:
    uv run python scripts/fetch_test_series.py                      # test_series/ -> data/raw/
    uv run python scripts/fetch_test_series.py --prefix train_series/<study_uid>/
"""

import argparse
import concurrent.futures as cf
import sys
import urllib.request
from pathlib import Path

from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsna_knee.constants import COMPETITION, RAW_DIR

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_archive_url() -> str:
    """Resolve the competition download-all endpoint to its signed GCS URL (valid ~days)."""
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}",
        headers={"Authorization": f"Bearer {TOKEN_PATH.read_text().strip()}"},
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(req)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
            return e.headers["Location"]
        raise SystemExit(f"Kaggle returned HTTP {e.code}; run `kaggle auth login` if it is 401.")
    raise SystemExit("Expected a redirect to the signed archive URL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="test_series/")
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    url = signed_archive_url()
    with RemoteZip(url) as z:
        members = [m for m in z.infolist() if m.filename.startswith(args.prefix) and not m.is_dir()]
    todo = [m for m in members if not ((args.out / m.filename).exists()
                                       and (args.out / m.filename).stat().st_size == m.file_size)]
    total_mb = sum(m.compress_size for m in todo) / 1e6
    print(f"{len(members)} members under {args.prefix}; {len(todo)} to fetch ({total_mb:.0f} MB compressed)",
          flush=True)

    def fetch(names: list[str]) -> int:
        # one RemoteZip per worker: the central directory is re-read once, then pure range reads
        with RemoteZip(url) as zw:
            for n in names:
                zw.extract(n, args.out)
        return len(names)

    chunks = [[m.filename for m in todo[i :: args.workers]] for i in range(args.workers)]
    done = 0
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for n in pool.map(fetch, [c for c in chunks if c]):
            done += n
            print(f"  {done}/{len(todo)} extracted", flush=True)
    print(f"done -> {args.out / args.prefix}")


if __name__ == "__main__":
    main()
