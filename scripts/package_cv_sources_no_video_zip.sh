#!/usr/bin/env bash
# Zip the CV repository in one pass (no rsync): no videos, no weight checkpoints.
# Uses: find (prune heavy dirs) | zip -@  so all depths are covered and disk is read once per file.
#
# Usage:
#   bash scripts/package_cv_sources_no_video_zip.sh [OUTPUT.zip]
#
# Environment:
#   CV_REPO_ROOT           — path to CV repo (default: parent of scripts/)
#   EXCLUDE_GIT=1          — omit .git
#   ZIP_LEVEL=N            — 0–9 (default 1 = fast); 0 = store only (fastest, largest)
#   PACK_SKIP_BSA_BUILD=1 — omit Block-Sparse-Attention/build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CV_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="$(basename "$ROOT")"
PARENT="$(cd "$(dirname "$ROOT")" && pwd)"
DEFAULT_OUT="$(dirname "$ROOT")/${BASE}_sources_no_video_${STAMP}.zip"
OUT="${1:-$DEFAULT_OUT}"
ZIP_LEVEL="${ZIP_LEVEL:-1}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: not a directory: $ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

echo "Zipping (find | zip -@, no staging): $PARENT/$BASE -> $OUT_ABS"

cd "$PARENT"

FIND=( find "$BASE" )

if [[ "${EXCLUDE_GIT:-0}" == "1" ]]; then
  FIND+=( \( -path "$BASE/.git" -o -path "$BASE/.git/*" \) -prune -o )
fi

if [[ "${PACK_SKIP_BSA_BUILD:-0}" == "1" ]]; then
  FIND+=( \( -path "$BASE/Block-Sparse-Attention/build" -o -path "$BASE/Block-Sparse-Attention/build/*" \) -prune -o )
fi

FIND+=(
  \( -path "$BASE/.cursor" -o -path "$BASE/.cursor/*" \) -prune -o
  \( -name '__pycache__' -type d \) -prune -o
  \( -name '.pytest_cache' -type d \) -prune -o
  \( -name '.mypy_cache' -type d \) -prune -o
  -type f
  ! \( -name '*.mp4' -o -name '*.MP4' -o -name '*.mov' -o -name '*.MOV' \
    -o -name '*.avi' -o -name '*.mkv' -o -name '*.webm' -o -name '*.m4v' \
    -o -name '*.pth' -o -name '*.PTH' -o -name '*.pt' -o -name '*.PT' \
    -o -name '*.safetensors' -o -name '*.ckpt' -o -name '*.CKPT' \
    -o -name '*.pyc' \)
  -print
)

"${FIND[@]}" | zip -rq@ "-$ZIP_LEVEL" "$OUT_ABS" -

echo "Done ($(du -h "$OUT_ABS" | cut -f1))"
