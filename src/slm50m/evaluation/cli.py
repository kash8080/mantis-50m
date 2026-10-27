"""Run every benchmark on a checkpoint and write summary.json with II (D-009, D-010).

    uv run slm-evaluate runs/main/ckpt/final_0034000.pt runs/main/evals/final --device cuda
"""

import argparse
import json
from pathlib import Path

from slm50m.config import TOKENIZER_DIR
from slm50m.evaluation.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="items per lm-eval task, for quick checks only")
    args = parser.parse_args()
    summary = evaluate_checkpoint(Path(args.checkpoint), Path(args.tokenizer_dir), Path(args.out_dir),
                                  args.device, args.limit)
    print(json.dumps({"scores": summary["scores"], "ii": summary["ii"]}, indent=2))
