"""Local CLI, useful for smoke tests and non-Modal training."""
from __future__ import annotations
import argparse
from common.config import TrainConfig
from training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    run_training(args.data, args.output, TrainConfig.load(args.config), resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
