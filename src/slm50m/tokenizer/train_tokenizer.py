"""Byte-level BPE tokenizer with one token per digit (D-018).

Pre-tokenizer is SmolLM2's: split every digit on its own, then GPT-2 byte-level with its regex.
llama.cpp already implements this split as its `smollm` pre-tokenizer (D-017).
"""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# id 0 is the document separator, eos and pad. The rest are reserved for SFT so the vocab never changes
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
EOS_TOKEN = SPECIAL_TOKENS[0]
VOCAB_SIZE = 16_384


def new_tokenizer() -> Tokenizer:
    """An untrained tokenizer with our pipeline: no normalizer, digits split, byte-level, no post-processor."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Train on an iterator of documents. vocab_size counts the special tokens too."""
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,  # added first, so <|endoftext|> gets id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, so nothing is ever unknown
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.token_to_id(EOS_TOKEN) == 0
    return tok


def save_tokenizer(tok: Tokenizer, out_dir: str | Path) -> Path:
    """Write tokenizer.json + tokenizer_config.json, the files transformers and llama.cpp read (D-017)."""
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS_TOKEN, pad_token=EOS_TOKEN)
    hf_tok.save_pretrained(out_dir)
    return out_dir


def load_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    """The raw `tokenizers` object, used by prep and training (fast batch encoding)."""
    return Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))


def load_hf_tokenizer(tokenizer_dir: str | Path):
    """Load the way the ArithMark script does. transformers 5.x ignores use_fast (D-017)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
