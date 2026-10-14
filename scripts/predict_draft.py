from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.inference import DEFAULT_CHECKPOINT, DraftPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next BP hero from a JSON state, without training data")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, help="JSON state file; omit to read stdin")
    args = parser.parse_args()
    try:
        source = args.input.read_text(encoding="utf-8-sig") if args.input else sys.stdin.read()
        result = DraftPredictor(args.checkpoint).predict(json.loads(source))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
