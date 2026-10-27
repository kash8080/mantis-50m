#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit (D-035) into _build/llama.cpp and build the tools the GGUF test uses.
# Needs cmake (brew install cmake). Training never needs this: it is only for the GGUF / quantization path (D-017),
# which must pass before the gate 5 main run.
#
#   bash scripts/setup_llama_cpp.sh
#   LLAMA_CPP_DIR=_build/llama.cpp uv run pytest tests/tokenizer -k gguf
#
# The script adds our tokenizer's chkhsh to get_vocab_base_pre in conversion/base.py (mapped to "smollm"). If the
# converter says "BPE pre-tokenizer was not recognized", the tokenizer changed: put its printed chkhsh in OUR_CHKHSH
set -euo pipefail

COMMIT=df9beac
DIR="$(cd "$(dirname "$0")/.." && pwd)/_build/llama.cpp"

if [ ! -d "$DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DIR"
fi
git -C "$DIR" checkout "$COMMIT"
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build "$DIR/build" -j --target llama-tokenize llama-quantize llama-cli llama-debug
# Our tokenizer uses SmolLM2's pre-tokenizer (D-018), so map its hash to "smollm". The hash comes from
# outputs/tokenizer: retrain the tokenizer and it changes (the converter then prints the new one)
OUR_CHKHSH=aa34ad5c40df88728f1e1e53a958ced7699ec112ff9977b20b2d63b1d6cb4c05
python3 - "$DIR/conversion/base.py" "$OUR_CHKHSH" <<'EOF'
import sys
from pathlib import Path

path, chkhsh = Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
if chkhsh not in text:
    anchor = '            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M\n            res = "smollm"\n'
    assert text.count(anchor) == 1, "smollm anchor not found: the pinned commit changed?"
    ours = f'        if chkhsh == "{chkhsh}":\n            # ref: SLM-50M tokenizer, SmolLM2 pre-tokenizer (D-018)\n            res = "smollm"\n'
    path.write_text(text.replace(anchor, anchor + ours))
    print("added the smollm hash line for our tokenizer")
EOF
# No pip install: the converter imports llama.cpp's own gguf-py from the checkout, and the rest of its deps
# (torch, transformers, numpy, sentencepiece, ...) are in uv.lock. Its requirements file would downgrade torch
# and transformers in this env
echo "built in $DIR. Run: LLAMA_CPP_DIR=$DIR uv run pytest tests/tokenizer -k gguf"
