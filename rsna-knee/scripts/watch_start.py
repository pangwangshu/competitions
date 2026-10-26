"""After a push: confirm the session really started by waiting for its first log line.

  python scripts/watch_start.py wangshu/rsna-knee-train [--minutes 20] [--pattern "linked|train "]

Kaggle shows RUNNING while a GPU session is still preparing inputs, and a session can hang
there for hours. The log stream works for sessions that are executing, so no matching line
within the deadline means the session should be stopped and pushed again. Exit 0 = started,
2 = no log by the deadline, 3 = the run already ended.
"""

import argparse
import re
import sys
import threading
import time


def read_lines(api, slug, seconds):
    lines = []

    def read():
        try:
            for e in api.kernels_logs_stream(slug):
                lines.append(str(e.get("data", e) if isinstance(e, dict) else e))
        except Exception as error:
            lines.append(f"__stream_error__ {type(error).__name__}: {error}")

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(seconds)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--pattern", default=r"linked \d+ cached|train \d+\s+val|\"run\":|started")
    args = ap.parse_args()
    from kaggle import api

    api.authenticate()
    deadline = time.time() + 60 * args.minutes
    while True:
        status = api.kernels_status(args.slug)
        name = getattr(status.status, "name", str(status.status)).upper()
        if name in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"}:
            print(f"{time.strftime('%H:%M:%S')} run ended: {name}")
            return 3
        lines = read_lines(api, args.slug, 45)
        hits = [line.strip() for line in lines if re.search(args.pattern, line)]
        errors = [line for line in lines if line.startswith("__stream_error__")]
        print(f"{time.strftime('%H:%M:%S')} {name}: {len(lines)} log lines, {len(hits)} matching"
              + (f"; {errors[0][:90]}" if errors else ""), flush=True)
        if hits:
            print("STARTED:", hits[0][:160])
            return 0
        if time.time() > deadline:
            print(f"NOT STARTED within {args.minutes:g} minutes: stop this session and push again")
            return 2
        time.sleep(90)


if __name__ == "__main__":
    sys.exit(main())
