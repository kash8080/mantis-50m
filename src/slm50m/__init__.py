"""SLM-50M: a sub-50M parameter language model pretrained from scratch.

Folders, in pipeline order:
  tokenizer/   train the BPE tokenizer (D-018)
  arith/       synthetic arithmetic generator and its proxy eval (D-011, D-024)
  data/        sources, benchmark filter, prep to token shards, mixture loader (D-022 to D-025)
  model/       the Qwen3-style decoder and its HF export (D-016, D-017, D-019)
  training/    loss, TST, optimizer, schedule, checkpoints, the training loop (D-026 to D-031)
  evaluation/  lm-eval + ArithMark wrapper and the Intelligence Index (D-009, D-010)

config.py holds the model presets and the run config.
"""
