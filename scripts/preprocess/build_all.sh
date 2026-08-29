#!/usr/bin/env bash
# Build the ordinary corpus: run every source processor into a staging tree,
# then publish it atomically to the output directory.
#
# Usage: scripts/preprocess/build_all.sh [--jobs N] [--output-dir DIR] [--keep-backup|--no-keep-backup]
#
# Heavy preprocessing must run on a pod, never on the dev box. The pod has no
# direct internet access: uv runs with --frozen --offline so a stale .venv
# fails fast instead of hanging on network timeouts. Run `uv sync` first if
# the lockfile changed. Run logs land in logs/preprocess/<timestamp>-<pid>/,
# never inside the published data tree.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$PROJECT_DIR"

jobs=3
output="$(realpath -m data/processed)"
keep_backup=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) jobs="$2"; shift 2 ;;
        --output-dir) output="$(realpath -m "$2")"; shift 2 ;;
        --keep-backup) keep_backup=true; shift ;;
        --no-keep-backup) keep_backup=false; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer" >&2
    exit 2
fi
case "$output" in
    "$PROJECT_DIR" | "$PROJECT_DIR"/) echo "output must be a child of the project root" >&2; exit 2 ;;
    "$PROJECT_DIR"/*) ;;
    *) echo "output directory must be a child of the project root: $output" >&2; exit 2 ;;
esac

processors=(
    process_crossad.py
    process_dada.py
    process_dcdetector.py
    process_easytsad.py
    process_langtime.py
    process_tsbad.py
)

# Newly audited sources are optional because WADI is access-controlled and the
# other corpora are downloaded separately. Once their raw marker exists they
# participate in the same staged build automatically.
optional_processors=(
    "process_aerca.py:data/raw/AERCA"
    "process_wadi.py:data/raw/WADI"
    "process_granite_tsfm.py:data/raw/GraniteTSFM/ZafNoo.csv"
    "process_gift_eval.py:data/raw/GiftEval"
)
for spec in "${optional_processors[@]}"; do
    script="${spec%%:*}"
    marker="${spec#*:}"
    if [[ -e "$marker" ]]; then
        processors+=("$script")
    else
        echo "skip optional source for $script (missing $marker)"
    fi
done

timestamp="$(date +%Y%m%d-%H%M%S)"
output_parent="$(dirname "$output")"
mkdir -p "$output_parent"
staging="$(mktemp -d "${output_parent}/.${output##*/}.staging-${timestamp}-XXXXXX")"
log_dir="$PROJECT_DIR/logs/preprocess/${timestamp}-$$"
mkdir -p "$log_dir"
echo "staging=$staging"
echo "logs=$log_dir"

fingerprint() {
    sha256sum pyproject.toml uv.lock scripts/preprocess/common.py \
        scripts/preprocess/process_*.py src/agentad/series/*.py
}
fingerprint > "$log_dir/build-inputs.sha256"
{
    echo "created_at=$(date --iso-8601=seconds)"
    echo "command=$0 $*"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    git status --short 2>/dev/null || true
} > "$log_dir/build-info.txt"

run_processor() {
    script="$1" staging="$2" log_dir="$3"
    log_path="$log_dir/${script%.py}.log"
    started=$SECONDS
    {
        printf 'command=uv run --frozen --offline python scripts/preprocess/%s --output-dir %s\n' \
            "$script" "$staging"
        uv run --frozen --offline python "scripts/preprocess/$script" --output-dir "$staging"
    } > "$log_path" 2>&1
    rc=$?
    printf '%s: rc=%s duration=%ss log=%s\n' \
        "$script" "$rc" "$((SECONDS - started))" "$log_path"
    return "$rc"
}
export -f run_processor

failed=0
printf '%s\n' "${processors[@]}" |
    xargs -P "$jobs" -I {} bash -c 'run_processor "$1" "$2" "$3"' _ \
        {} "$staging" "$log_dir" || failed=1
if [[ "$failed" -ne 0 ]]; then
    echo "build failed; staging retained at $staging" >&2
    exit 1
fi

if ! fingerprint | cmp -s - "$log_dir/build-inputs.sha256"; then
    fingerprint > "$log_dir/build-inputs-after.sha256"
    echo "build inputs changed during the build; staging retained at $staging" >&2
    exit 1
fi

backup="${output}.before-${timestamp}-$$"
had_output=false
if [[ -e "$output" || -L "$output" ]]; then
    had_output=true
    mv "$output" "$backup"
fi
if mv "$staging" "$output"; then
    if $had_output && ! $keep_backup; then
        rm -rf "$backup"
        echo "published=$output backup=removed"
    elif $had_output; then
        echo "published=$output backup=$backup"
    else
        echo "published=$output"
    fi
else
    if $had_output && [[ ! -e "$output" ]]; then
        mv "$backup" "$output"
    fi
    echo "publish failed; staging retained at $staging" >&2
    exit 1
fi
