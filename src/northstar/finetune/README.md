# Fine-tune (Phase 5) — owner: Neeraj (reference build: Kawtar)

A small encoder (`xlm-roberta-base`) fine-tuned to produce `RouterOutput` (request type + escalation) directly from the chat text, trained on the 130 training chats and validated on 30, as an alternative to the prompted router. Compared against the small (Haiku) and large (Sonnet) prompted baselines on the same 40 held-out test chats.

Developed and trained interactively in a Kaggle notebook (free T4 GPU); `train_router.py` here is the same pipeline as a standalone, reproducible script.

## Files

- `train_router.py` — the training/evaluation pipeline. Loads the data, builds the model, trains, evaluates once on test. Also runs the learning-rate sweep and the stability check from the CLI.
- `hyperparameters.json` — final settings, base-model search, training cost, and the stability-check summary, all in one record.
- `MODEL_CARD.md` — what the model is, why this base model, where the checkpoint lives, and its known limitation (run-to-run accuracy instability on this small a dataset).
- `results/` — logged outputs, not regenerated on every run:
  - `base_model_comparison.csv` — 4 candidate encoders, same pipeline, same data.
  - `learning_rate_comparison.csv` — 1e-5 / 2e-5 / 5e-5 on xlm-roberta-base.
  - `stability_runs.csv` — 5 identical reruns of the final settings, logged individually.
  - `final_comparison_table.csv` — fine-tuned model vs. Haiku vs. Sonnet on the 40 test chats.

## Reproducing

```bash
pip install -r requirements.txt torch transformers scikit-learn

# Final model (what results/final_comparison_table.csv used)
python -m northstar.finetune.train_router --model xlm-roberta-base --lr 2e-5 --epochs 30 \
    --save src/northstar/finetune/xlmr_router_best.pt

# Learning-rate sweep
python -m northstar.finetune.train_router --model xlm-roberta-base --lr-sweep 1e-5 2e-5 5e-5

# Stability check (5 identical reruns)
python -m northstar.finetune.train_router --model xlm-roberta-base --lr 2e-5 --stability-runs 5
```

Defaults to `data/conversations/conversations.csv`; pass `--data` to point at another copy of the dataset.

## Headline result

Fine-tuning did not beat either prompted baseline on type accuracy, and the gap itself is noisy: the fine-tuned model's own accuracy swings 57.5%-72.5% (mean 66.0%) across identical reruns, from random init on only 130 training examples, while Haiku and Sonnet sit at 75.0% and 77.5%. Fine-tuning wins by a wide margin on latency (~9.5ms vs 958-2132ms) and cost ($0/chat vs $0.0004-$0.0055/chat). Full writeup: report Page 4 and Appendix; numbers in `results/` and `hyperparameters.json`.

No secrets are used anywhere in this folder or `train_router.py`.
