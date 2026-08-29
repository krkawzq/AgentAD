"""Download official pretrained checkpoints into pretrained/.

Covers the three methods with published weights:

- TSPulse:  Hugging Face snapshot of ``ibm-granite/granite-timeseries-tspulse-r1``
- DADA:     the published HF-format checkpoint vendored in ``forks/DADA/DADA/``
- Time-RCD: ``thu-sail-lab/Time-RCD`` best-model checkpoints (uni/multi)

The Hugging Face downloads need outbound network access; run with the
workspace proxy exported (``labpon``) or an ``HF_ENDPOINT`` mirror that is
reachable from the current host.

Usage:
    python download_pretrained.py [--output-dir DIR] [--only tspulse|dada|time-rcd]
"""

import argparse
import shutil
import sys
from pathlib import Path

TSPULSE_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
TIME_RCD_REPO = "thu-sail-lab/Time-RCD"
TIME_RCD_FILES = {
    "uni": "best_model/pretrain_checkpoint_best_uni.pth",
    "multi": "best_model/pretrain_checkpoint_best_multi.pth",
}
DADA_SOURCE = Path(__file__).resolve().parents[2] / "forks" / "DADA" / "DADA"


def download_tspulse(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = out_dir / "tspulse"
    for name in ("config.json", "model.safetensors"):
        dest = target / name
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        print(f"downloading TSPulse file: {name}")
        hf_hub_download(repo_id=TSPULSE_REPO, filename=name, local_dir=target)
    print(f"TSPulse weights -> {target}")


def download_dada(out_dir: Path) -> None:
    target = out_dir / "dada"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("pytorch_model.bin", "config.json"):
        source = DADA_SOURCE / name
        dest = target / name
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        if not source.exists():
            sys.exit(f"DADA checkpoint file missing in fork: {source}")
        print(f"copying DADA checkpoint: {source} -> {dest}")
        shutil.copy2(source, dest)


def download_time_rcd(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = out_dir / "time_rcd"
    for variant, repo_file in TIME_RCD_FILES.items():
        variant_dir = target / variant
        dest = variant_dir / Path(repo_file).name
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        print(f"downloading Time-RCD {variant}: {repo_file}")
        downloaded = Path(
            hf_hub_download(
                repo_id=TIME_RCD_REPO,
                filename=repo_file,
                local_dir=variant_dir,
            )
        )
        if downloaded != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), dest)
            # drop the empty repo-layout directory left behind
            leftover = variant_dir / Path(repo_file).parent
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()
        print(f"Time-RCD {variant} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official pretrained checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained"),
        help="output directory (default: pretrained/)",
    )
    parser.add_argument(
        "--only",
        choices=["tspulse", "dada", "time-rcd"],
        default=None,
        help="download only one method's weights",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        "tspulse": download_tspulse,
        "dada": download_dada,
        "time-rcd": download_time_rcd,
    }
    selected = [args.only] if args.only else list(tasks)
    for name in selected:
        tasks[name](args.output_dir)


if __name__ == "__main__":
    main()
