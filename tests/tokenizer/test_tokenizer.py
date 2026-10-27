"""Gate 0: digit splitting, the space before a number, round trips, ArithMark ids (D-017, D-018), GGUF path."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from slm50m.config import PRESETS, TOKENIZER_DIR
from slm50m.tokenizer.train_tokenizer import load_hf_tokenizer, load_tokenizer
from tests.conftest import ENGLISH


def test_digits_and_space_before_numbers(tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    assert tok.token_to_id("<|endoftext|>") == 0
    assert tok.encode("12345").tokens == ["1", "2", "3", "4", "5"]
    # the space before a number is its own token, so " 63" tokenized alone matches how it looks mid-text
    assert tok.encode(" 63").tokens == ["Ġ", "6", "3"]
    assert tok.encode("had 63 cards").tokens[-4:-1] == ["Ġ", "6", "3"]


def test_round_trip(tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    for text in ENGLISH + ["   leading spaces", "tabs\tand\nnewlines\n\n", "emoji 🙂 and ½ and 漢字", ""]:
        assert tok.decode(tok.encode(text).ids) == text


def test_arithmark_loader_gives_training_ids(tokenizer_dir, arithmark):
    """The ArithMark script loads with AutoTokenizer(use_fast=False) and tokenizes context and ending
    separately. That must give exactly the ids our prep uses, on every item."""
    ours = load_tokenizer(tokenizer_dir)
    hf = load_hf_tokenizer(tokenizer_dir)
    for row in arithmark:
        for text in [row["ctx"], *row["endings"]]:
            assert hf.encode(text, add_special_tokens=False) == ours.encode(text, add_special_tokens=False).ids


LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR")


@pytest.mark.skipif(not LLAMA_CPP, reason="set LLAMA_CPP_DIR to a llama.cpp build (scripts/setup_llama_cpp.sh)")
def test_gguf_path(tmp_path):
    """The main shape with random weights and the real gate 1 tokenizer converts to GGUF, tokenizes like ours,
    gives our fp32 logits in llama.cpp, and quantizes (D-012, D-017). The real tokenizer, not the test fixture:
    llama.cpp's smollm hash line matches its vocab (D-039). Needed before gate 5."""
    from slm50m.model.export import export
    from tests.model.test_model import randomized

    llama = Path(LLAMA_CPP)
    tokenizer_dir = TOKENIZER_DIR
    model = randomized(PRESETS["main"])  # the real init would make every block the identity
    hf_dir = export(model, tmp_path / "hf", tokenizer_dir)
    f32 = tmp_path / "model-f32.gguf"
    conv = subprocess.run([sys.executable, llama / "convert_hf_to_gguf.py", hf_dir, "--outfile", f32, "--outtype", "f32"],
                          capture_output=True, text=True, check=False)
    assert conv.returncode == 0, conv.stderr[-3000:]  # "not recognized" means the smollm hash line is missing

    ours = load_tokenizer(tokenizer_dir)
    for text in ENGLISH + [" 63", "12345", "Sam had 43 rubber bands."]:
        out = subprocess.run([llama / "build/bin/llama-tokenize", "-m", f32, "-p", text, "--ids", "--log-disable",
                              "--no-bos"], capture_output=True, text=True, check=True).stdout
        assert eval(out.strip().splitlines()[-1]) == ours.encode(text).ids, text

    # llama.cpp's last-position logits on the CPU vs ours, on the tokens it fed (it writes them out too)
    out_dir = tmp_path / "logits"
    subprocess.run([llama / "build/bin/llama-debug", "-m", f32, "-p", "Sam had 43 rubber bands. He gave 17 to",
                    "--save-logits", "--logits-output-dir", out_dir, "-ngl", "0", "-c", "64",
                    "-ctk", "f32", "-ctv", "f32", "-fa", "off"],  # all fp32: the defaults (f16 flash attention) add ~2e-3
                   capture_output=True, check=True)
    ids = torch.from_numpy(np.fromfile(out_dir / "llamacpp-model-f32-tokens.bin", dtype=np.int32).astype(np.int64))
    theirs = torch.from_numpy(np.fromfile(out_dir / "llamacpp-model-f32.bin", dtype=np.float32))
    assert ids.tolist() == ours.encode("Sam had 43 rubber bands. He gave 17 to").ids  # no BOS added
    with torch.no_grad():
        logits = model(ids[None])[0, -1]
    # ~3.5e-6 measured (27 Oct 2026), rounding only. A real mismatch (RoPE, GQA, q/k norm) is order 1
    assert (logits - theirs).abs().max() / logits.abs().max() < 1e-4

    for quant in ("Q8_0", "Q4_K_M"):
        subprocess.run([llama / "build/bin/llama-quantize", f32, tmp_path / f"model-{quant}.gguf", quant],
                       capture_output=True, check=True)
