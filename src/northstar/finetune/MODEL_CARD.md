# Model card — fine-tuned router

**Task:** classify a customer chat into one of 10 request types and decide whether it needs human escalation — the same job the prompted router (`northstar.agent.router`) does, same `RouterOutput` contract, so it can be swapped in behind the same plug.

**Base model:** `xlm-roberta-base` (278,052,107 parameters), from Hugging Face `transformers`.

**Architecture:** the pretrained encoder, unchanged, plus two linear heads on the `[CLS]` token's final hidden state — a 10-way head for request type, a single-logit head for escalation. Dropout 0.2 before both heads.

**Training data:** 130 training chats, 30 validation chats, 40 test chats (see `config/settings.yaml -> paths.conversations`), grouped by customer so no customer appears in more than one split. The test split was evaluated exactly once, at the end.

**Why this base model:** four encoders were tried with the identical pipeline (`train_router.py`) — see `results/base_model_comparison.csv`:

| model | params | type accuracy |
|---|---|---|
| distilbert-base-multilingual-cased | 135M | 45.0% |
| Geotrend/distilbert-base-en-fr-cased | 69M | 25.0% |
| **xlm-roberta-base** | 278M | 62.5% |
| microsoft/mdeberta-v3-base | — | excluded: unrecoverable CUDA OOM at batch_size=8, from its disentangled attention mechanism (architectural, not a performance result) |

xlm-roberta-base won clearly and was carried forward as the final model.

**Checkpoint:** not committed to this repository — the trained weights (`xlmr_router_best.pt`, ~1.1 GB) live in the Kaggle notebook's output and are too large for a normal git repo. To reproduce it:

```bash
python -m northstar.finetune.train_router \
    --model xlm-roberta-base --lr 2e-5 --epochs 30 \
    --save src/northstar/finetune/xlmr_router_best.pt
```

Training is free (Kaggle T4) or under USD 0.04 on a paid cloud GPU, and takes about 3.5 minutes per run — retraining from scratch is cheaper than storing or transferring the checkpoint.

**Known limitation — run-to-run instability:** with only 130 training examples, the same settings retrained from scratch land anywhere from 57.5% to 72.5% type accuracy depending on random initialization and batch order (mean 66.0%, std 5.5 points over 5 runs — see `results/stability_runs.csv`). Any single reported number for this model should be read as one draw from that range, not a fixed property of the approach. This is also why the learning-rate choice (`results/learning_rate_comparison.csv`) is not a clean, reproducible ranking: reruns of the same 1e-5/2e-5/5e-5 comparison have picked different winners.

**Where it sits vs. the prompted baselines:** see `results/final_comparison_table.csv` and the report (Page 4). On the comparison-table run: 65.0% type accuracy vs Haiku's 75.0% and Sonnet's 77.5% — but given the instability above, the honest range for the fine-tuned model is roughly 57.5-72.5%, still below both baselines in every run observed. It wins by orders of magnitude on latency (~9.5ms vs 958-2132ms) and on cost ($0/chat after training vs $0.0004-$0.0055/chat).
