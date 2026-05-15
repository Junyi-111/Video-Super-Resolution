#!/usr/bin/env bash
# Pack the CV repository in one pass (no rsync copy): no videos, no weight checkpoints.
# Excludes: mp4/mov/avi/mkv/webm/m4v, plus *.pth *.pt *.safetensors *.ckpt, caches, optional .git.
#
# Usage:
#   bash scripts/package_cv_sources_no_video.sh [OUTPUT_TAR_GZ]
#
# Environment:
#   CV_REPO_ROOT        — path to CV repo (default: parent of scripts/)
#   EXCLUDE_GIT=1       — omit .git from the archive
#   USE_PIGZ=1          — use pigz parallel gzip if installed (faster than gzip)
#   PACK_SKIP_BSA_BUILD=1 — omit Block-Sparse-Attention/build if present
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CV_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="$(basename "$ROOT")"
PARENT="$(cd "$(dirname "$ROOT")" && pwd)"
DEFAULT_OUT="$(dirname "$ROOT")/${BASE}_sources_no_video_${STAMP}.tar.gz"
OUT="${1:-$DEFAULT_OUT}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: not a directory: $ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

TAR_EXCLUDES=(
  --exclude="${BASE}/.cursor"
  --exclude='*.mp4'
  --exclude='*.MP4'
  --exclude='*.mov'
  --exclude='*.MOV'
  --exclude='*.avi'
  --exclude='*.mkv'
  --exclude='*.webm'
  --exclude='*.m4v'
  --exclude='*.pyc'
  --exclude='__pycache__'
  --exclude='.pytest_cache'
  --exclude='.mypy_cache'
  --exclude='*.pth'
  --exclude='*.PTH'
  --exclude='*.pt'
  --exclude='*.PT'
  --exclude='*.safetensors'
  --exclude='*.ckpt'
  --exclude='*.CKPT'
)

if [[ "${PACK_SKIP_BSA_BUILD:-0}" == "1" ]]; then
  TAR_EXCLUDES+=(--exclude="${BASE}/Block-Sparse-Attention/build")
fi

if [[ "${EXCLUDE_GIT:-0}" == "1" ]]; then
  TAR_EXCLUDES+=(--exclude="${BASE}/.git")
fi

echo "Packing (one pass, no staging): $PARENT/$BASE -> $OUT_ABS"

if [[ "${USE_PIGZ:-0}" == "1" ]] && command -v pigz >/dev/null 2>&1; then
  tar -C "$PARENT" "${TAR_EXCLUDES[@]}" -cf - "$BASE" | pigz -p "${PIGZ_PROCS:-$(nproc 2>/dev/null || echo 4)}" >"$OUT_ABS"
else
  tar -C "$PARENT" "${TAR_EXCLUDES[@]}" -czf "$OUT_ABS" "$BASE"
fi

echo "Done ($(du -h "$OUT_ABS" | cut -f1))"
