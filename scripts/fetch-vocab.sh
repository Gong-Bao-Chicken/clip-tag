#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://raw.githubusercontent.com/Jsakkos/lightroom-autotag/main/src/lr_autotag/Foundation%20List%202.0.1.txt"
OUT="$ROOT/assets/vocab/default_labels.txt"
mkdir -p "$(dirname "$OUT")"
curl -sL "$URL" | python3 -c "
import sys
from pathlib import Path
out = Path(sys.argv[1])
lines = sys.stdin.read().splitlines()
labels = sorted({ln.strip().lower() for ln in lines if ln.strip() and not ln.strip().startswith('[') and not ln.strip().startswith('~')})
out.write_text('\n'.join(labels) + '\n')
print(f'Wrote {len(labels)} labels to {out}')
" "$OUT"
