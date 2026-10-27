"""Export a checkpoint to a Qwen3ForCausalLM folder (D-017).

    uv run slm-export-hf runs/main/ckpt/final_0034000.pt _build/export/main
"""

import argparse

from slm50m.config import TOKENIZER_DIR
from slm50m.model.export import export, load_checkpoint_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR))
    args = parser.parse_args()
    model = load_checkpoint_model(args.checkpoint)
    print(f"{model.n_params():,} params -> {export(model, args.out_dir, args.tokenizer_dir)}")
