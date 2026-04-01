#!/usr/bin/env python3
"""
Pipeline runner for Petri + ASFormer.

Runs selected steps:
  discover -> train -> predict -> eval

Example:
  python run_pipeline.py --asformer-root "/Users/yaelreina/ASFormer-main" --steps all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"\n>>> ({cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run full Petri + ASFormer pipeline")
    parser.add_argument(
        "--petri-root",
        default=str(script_dir),
        help="Path to Petri project (default: this script directory)",
    )
    parser.add_argument(
        "--asformer-root",
        default=str(Path.home() / "ASFormer-main"),
        help="Path to ASFormer-main project",
    )
    parser.add_argument("--dataset", default="breakfast")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root path (contains mapping.txt, groundTruth, features, splits)")
    parser.add_argument("--split", default="1")
    parser.add_argument("--petri-mode", default="discover", choices=["discover", "load"])
    parser.add_argument(
        "--petri-net-file",
        default="",
        help="Path to PNML file. In load mode: existing file to load. In discover mode: optional save path.",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps: discover,train,predict,eval or 'all'",
    )
    parser.add_argument(
        "--weight-disagreement",
        default="1.5",
        help="Pass-through to main.py (used in train/predict with --use_petri)",
    )
    args = parser.parse_args()

    petri_root = Path(args.petri_root).expanduser().resolve()
    asformer_root = Path(args.asformer_root).expanduser().resolve()

    if args.steps == "all":
        steps = ["discover", "train", "predict", "eval"]
    else:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    valid = {"discover", "train", "predict", "eval"}
    unknown = [s for s in steps if s not in valid]
    if unknown:
        raise SystemExit(f"Unknown step(s): {unknown}. Valid: {sorted(valid)}")

    py = sys.executable

    dataset_root_resolved = (
        args.dataset_root.strip()
        if args.dataset_root.strip()
        else str(asformer_root / "data" / args.dataset)
    )

    train_petri_mode = args.petri_mode
    if "discover" in steps and args.petri_net_file.strip():
        train_petri_mode = "load"

    if "discover" in steps:
        discover_cmd = [
            py,
            "petri_net_inductive_miner.py",
            "--dataset-root",
            dataset_root_resolved,
            "--dataset",
            args.dataset,
        ]
        if args.petri_net_file:
            discover_cmd.extend(
                ["--output-pnml", str(Path(args.petri_net_file).expanduser().resolve())]
            )
        run_cmd(discover_cmd, cwd=petri_root)

    if "train" in steps:
        train_cmd = [
            py,
            "main.py",
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--dataset_root",
            dataset_root_resolved,
            "--use_petri",
            "--petri_path",
            str(petri_root),
            "--weight_disagreement",
            args.weight_disagreement,
            "--petri_mode",
            train_petri_mode,
        ]
        if args.petri_net_file:
            train_cmd.extend(["--petri_net_file", str(Path(args.petri_net_file).expanduser())])
        run_cmd(
            train_cmd,
            cwd=asformer_root,
        )

    if "predict" in steps:
        predict_cmd = [
            py,
            "main.py",
            "--action",
            "predict",
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--dataset_root",
            dataset_root_resolved,
            "--use_petri",
            "--petri_path",
            str(petri_root),
            "--weight_disagreement",
            args.weight_disagreement,
            "--petri_mode",
            train_petri_mode,
        ]
        if args.petri_net_file:
            predict_cmd.extend(["--petri_net_file", str(Path(args.petri_net_file).expanduser())])
        run_cmd(
            predict_cmd,
            cwd=asformer_root,
        )

    if "eval" in steps:
        run_cmd(
            [py, "eval.py", "--dataset", args.dataset, "--split", args.split],
            cwd=asformer_root,
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
