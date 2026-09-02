#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -d results ]; then
  echo "results/ not found" >&2
  exit 1
fi
rm -f results.zip
zip -r results.zip results
ls -lh results.zip
