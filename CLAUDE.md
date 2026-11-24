# Northstar Motors — Fine-Tuning & Model Comparison (Member 3)

Context for Claude Code picking this up. Read this before touching anything. If you're starting a new piece of work here (LoRA experiment or otherwise), this file is the ramp-up — the actual numbers and artifacts live in the files it points to, not duplicated here.

## Project

MMAI 894 course project. "Northstar Motors" is a fictional dealership; the team is building a multi-agent customer-service system. The repo this work merges into is `Salikawtar/northstar-agent` (team repo) — this repo (`neerajmehla28/northstar-finetuning`) is a staging area, not the final destination; see "Status / where things stand" below.

My role (Member 3): fine-tune a small model to do the **router**'s job — classify an incoming customer chat into one of 10 request types and decide whether it needs human escalation — and compare it against the team's existing prompted baselines (small model = Claude Haiku, large model = Claude Sonnet) on accuracy, latency, and cost. The router is the first thing every chat passes through, so its speed/cost matter as much as its accuracy.

Rubric requirements (from the assignment) for this piece: pick and justify a narrow task, document the base model and method, train only on the official train/val split, record hyperparameters/epochs/cost, implement all three baselines comparably (same input/output schema, same test cases), coordinate with Member 4 for held-out results, explain where fine-tuning wins/loses, evaluate when it's worth it at volume, and report a negative result clearly if prompting wins.

## What's been done

**Task & data:** 10-way request type + binary escalation classification. 130 train / 30 validation / 40 test chats, grouped by customer so no customer crosses splits, test touched exactly once. Data file: `conversations_fixed.csv` (team's official dataset — **verify with Member 1 this is the final labeled version**, see Open items below).

**Base model search:** tried 4 encoders with an identical pipeline — DistilBERT-multilingual (45.0% type accuracy), Geotrend en-fr (25.0%), **XLM-RoBERTa-base (62.5%, winner)**, mDeBERTa-v3-base (excluded: unrecoverable CUDA OOM from its attention mechanism, not a performance result). Full detail: `src/northstar/finetune/MODEL_CARD.md`.

**Final model:** `xlm-roberta-base`, 278,052,107 params, shared encoder + two linear heads on `[CLS]` (10-way type, binary escalation). Hyperparameters: batch size 16, lr 2e-5 (AdamW), CrossEntropyLoss + BCEWithLogitsLoss summed, max_length 256, 30 epochs checkpointed on best validation loss. Full record: `src/northstar/finetune/hyperparameters.json`.

**Learning-rate comparison:** tested 1e-5, 2e-5, 5e-5. Result was **inconsistent across different full reruns** — sometimes 2e-5 won, sometimes it tied with 1e-5, sometimes 5e-5 won. Kept 2e-5 as the standard default, not because any single run proved it best. Full numbers: `src/northstar/finetune/results/learning_rate_comparison.csv`.

**Critical finding — run-to-run instability:** retrained the exact same model/settings 5 times from scratch. Type accuracy swung **57.5% to 72.5%** (mean 66.0%, std 5.5 points) purely from random init and batch order, because 130 training examples is small. This is reported as the key negative-result finding, not hidden. Numbers: `src/northstar/finetune/results/stability_runs.csv`.

**Three-way comparison (one specific run, the one in the report table):**

| Model | Type acc | Macro F1 | Escalation acc | EN / FR acc | Latency (median) | Cost/chat |
|---|---|---|---|---|---|---|
| Fine-tuned XLM-R | 65.0% | 0.650 | 82.5% | 65.4% / 64.3% | 9.5 ms | $0 |
| Claude Haiku (small, prompted) | 75.0% | 0.735 | 92.5% | 69.2% / 85.7% | 2,132 ms | $0.000364 |
| Claude Sonnet (large, prompted) | 77.5% | 0.772 | 92.5% | 76.9% / 78.6% | 958 ms | $0.005546 |

Full table: `src/northstar/finetune/results/final_comparison_table.csv`. Fine-tuning **loses on accuracy** (and that gap itself is noisy — see stability finding above), **wins big on latency and cost**.

**Break-even analysis:** fine-tuning beats Haiku economically below ~$0.0040/misroute, beats Sonnet below ~$0.048/misroute (Sonnet's much higher per-chat cost gives it more room). Training cost is trivial (~$0.04/run, 3.5 min on a free Kaggle T4), so volume almost never matters — the real variable is how costly a misrouted chat actually is, which is Member (economics)'s input.

**Deliverables in this repo** (`src/northstar/finetune/`):
- `train_router.py` — the full pipeline as a reproducible CLI script (not just notebook cells): data loading, `RouterClassifier` model, `train_and_evaluate`, CLI flags for a single run / LR sweep / stability check.
- `hyperparameters.json`, `MODEL_CARD.md`, `README.md`, `results/*.csv` — as above.
- `dealer-chatbot-router-model-tuning.ipynb` — the final Kaggle notebook, run end-to-end with 0 errors. Source of truth for every number above (all outputs, including charts, are embedded in the file).

**Not included, deliberately:** the trained checkpoint (`xlmr_router_best.pt`, ~1.1GB) — too large for git, cheap to reproduce (see command in MODEL_CARD.md), not worth storing.

## Status / where things stand

- This repo (`neerajmehla28/northstar-finetuning`) is a **staging repo**, pushed here only because this GitHub account didn't have push access to the team repo (`Salikawtar/northstar-agent`) at the time. See this repo's own `README.md` for the exact merge commands.
- **Not yet merged into the team repo.** That still needs to happen — either get collaborator access on `Salikawtar/northstar-agent`, or hand this off to a teammate who has it.
- **Member 4 (Libo) coordination not done.** The rubric wants held-out accuracy/latency/cost confirmed with/by Member 4. Nobody outside this work has touched the eval yet — send Libo `results/final_comparison_table.csv` plus the per-chat result CSVs.
- **Dataset version unconfirmed.** Ask Member 1 whether `conversations_fixed.csv` (used for all the numbers above) is the final labeled version, or whether labels were still being finished when this was pulled.
- Report Page 4 (one-pager) and Appendix content were already drafted in an earlier session — ask the user if they still need that text, it may already be in the report doc.

## What's next

**LoRA experiment** (about to start): freeze the pretrained `xlm-roberta-base` weights, train small low-rank adapter matrices (rank 8, alpha 16, targeting the `query`/`value` attention projections) plus the two classification heads, instead of full fine-tuning. Rationale worth testing: full fine-tuning trains all 278M params on only 130 examples, which is the likely cause of the instability finding above; LoRA trains roughly 1% as many parameters, which is a reasonable hypothesis for *less* run-to-run variance, not just a cheaper run. Plan:
1. Add a `peft`-based LoRA wrapper around the encoder in a variant of `train_and_evaluate` (keep data loading, heads, and eval identical to the full fine-tune, for a clean comparison).
2. Run it once, then — consistent with how the full fine-tune was evaluated — repeat the stability check (5 reruns) so variance can be compared directly, not just accuracy.
3. Add a 4th row/column set to the comparison table and write up accuracy/latency/cost/variance vs. the full fine-tune and the two prompted baselines.

**Other planned steps:** the user mentioned "a few other steps" beyond LoRA without specifying them yet — ask what those are before assuming scope.

## Conventions to keep

- No hardcoded numbers in comparison tables — read from saved JSON/CSV files or compute live from DataFrames, not typed literals (this was a repeated bug caught twice in an earlier session).
- Report instability honestly — don't collapse the 5-run range into one cherry-picked number in prose.
- Keep the trained checkpoint out of git; document reproduction steps instead.
- The user wants plain, direct, succinct writing — no LLM-style hedging or filler.
