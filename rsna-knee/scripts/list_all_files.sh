#!/bin/bash
# Paginate the full Kaggle competition file listing with resume support.
# Cache: data/filelist_pages/page_N.csv (first line holds the next-page token).
# Transient auth blips (rate-limit masquerading as 401) are retried with backoff.
export PATH="$HOME/.local/bin:$PATH"
COMP=rsna-knee-abnormality-detection
CACHE=data/filelist_pages
mkdir -p "$CACHE"

page=$(ls "$CACHE" 2>/dev/null | sed -n 's/^page_\([0-9]*\)\.csv$/\1/p' | sort -n | tail -1)
if [ -n "$page" ]; then
  token=$(head -1 "$CACHE/page_${page}.csv" | sed -n 's/^#token: //p')
  page=$((page+1))
else
  page=1; token=""
fi

fetch_page() {
  if [ -z "$1" ]; then
    kaggle competitions files $COMP --page-size 200 -v 2>/dev/null
  else
    kaggle competitions files $COMP --page-size 200 -v --page-token "$1" 2>/dev/null
  fi
}

while : ; do
  out=""
  ok=""
  for attempt in 1 2 3 4 5; do
    out=$(fetch_page "$token")
    if [ -n "$out" ] && ! echo "$out" | grep -q "Authentication required"; then
      ok=1; break
    fi
    # empty or auth-flavoured response — usually a transient throttle
    wait=$((attempt * 60))
    echo "attempt $attempt failed at page $page (empty/auth), backing off ${wait}s"
    sleep $wait
  done
  [ -z "$ok" ] && { echo "giving up at page $page — rerun this script to resume"; break; }

  newtoken=$(echo "$out" | sed -n 's/^Next Page Token = //p')
  echo "$out" | grep -v '^Next Page Token' > "$CACHE/page_${page}.csv.tmp"
  { echo "#token: $newtoken"; cat "$CACHE/page_${page}.csv.tmp"; } > "$CACHE/page_${page}.csv"
  rm -f "$CACHE/page_${page}.csv.tmp"
  echo "page $page: $(($(wc -l < "$CACHE/page_${page}.csv") - 1)) files"
  token="$newtoken"
  [ -z "$token" ] && { echo "LISTING COMPLETE"; break; }
  page=$((page+1))
  sleep 1
done
