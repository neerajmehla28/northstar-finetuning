# Fine-tuning and model comparison tasks tracker

> **Internal working file.** Decide whether to keep or remove it before the repository is shared with the evaluators.

**Main idea:** every task for Member 3 (fine-tuning and model comparison, 8 marks), what it is, where it feeds into the project, and whether it is done. **Owner: Neeraj.** The fine-tuned model replaces the router behind the same "plug" (the `RouterOutput` contract), so nothing else in the agent changes.

Status values: **Done**, **Decided**, **Partly built**, **To do**.

| # | Task | What it is | Where it is used | Status |
|---|---|---|---|---|
| 1 | Narrow task | Classify the request type and decide whether a person must take over (the router's job) | Report page 4; runs on every chat, so speed and cost matter | Decided (project outline) |
| 2 | Base model and method | Pick a small base model and a training method, and say why | Report page 4 | Done: tested 4 encoders with the same pipeline, xlm-roberta-base won (see `src/northstar/finetune/results/base_model_comparison.csv`); mdeberta-v3-base excluded (unrecoverable CUDA OOM) |
| 3 | Training data | The 130 training chats as input and output pairs (chat in, `RouterOutput` JSON out); the 30 validation chats for tuning; never the 40 test chats | Training run | Done |
| 4 | Training run | Train and record model version, hyperparameters, epochs, settings and training cost | Report page 4 and appendix; training cost for Abhi | Done: `src/northstar/finetune/hyperparameters.json`; training cost under USD 0.04/run (free on Kaggle T4) |
| 5 | Plug into the agent | The fine-tuned model answers in the `RouterOutput` format, chosen in `config/settings.yaml` | End-to-end agent with the fine-tuned router | Done: `config/settings.yaml -> models.finetuned_router`; checkpoint not committed (see MODEL_CARD.md), reproducible via `train_router.py` |
| 6 | Small prompted baseline | Claude Haiku 5.5 with the router prompt | Comparison table | Done on test: prompt v3, 75.0% type accuracy, 100% escalation recall, $0.000364 per chat |
| 7 | Large prompted baseline | Claude Sonnet 5.5 with the same router prompt | Comparison table | Done on test: 77.5% type accuracy, 94.4% escalation recall, $0.005546 per chat |
| 8 | Same task for all three | Same input, same output format, same test chats | Fair comparison (brief requirement) | Done: all three went through the same router evaluation on the same 40 test chats |
| 9 | Metrics | Macro F1 and per-class accuracy for the request type, escalation recall, latency, cost | Comparison table; outline promise | Done: type accuracy, macro F1, escalation accuracy/recall/precision, EN/FR accuracy, latency, cost all in `results/final_comparison_table.csv` |
| 10 | Test-set comparison | The three routers on the 40 test chats, run once, with Libo | Report page 4 results table | Done: `results/final_comparison_table.csv` |
| 11 | Where it wins and loses | For example short English chats versus French and multi-ask chats | Report page 4 interpretation | Done: fine-tuning loses on accuracy (and the gap is noisy, not fixed — see stability check), wins big on latency and cost; EN/FR breakdown in the comparison table |
| 12 | When fine-tuning pays off | Monthly volume where the training cost is paid back by cheaper calls (with Abhi) | Report pages 4 and 7 | Done: break-even analysis vs. Haiku and Sonnet separately, as a function of misroute cost (report Step 29 / appendix) |
| 13 | Honest negative result | With about 16 training chats per type the fine-tune may lose; if so, say why and estimate the data size where it would win | Report page 4 (brief: "report a negative result clearly") | Done: fine-tuning lost on accuracy; also found the result itself is unstable run to run (5-run stability check, `results/stability_runs.csv`), reported as the negative result rather than cherry-picking one run |
| 14 | Scripts and logs in the repo | Training script, settings, model reference, training logs; no keys | GitHub submission; reproducibility | Done: `src/northstar/finetune/` (`train_router.py`, `hyperparameters.json`, `MODEL_CARD.md`, `results/`) |
| 15 | Report page 4 and appendix | Task, base model, splits, training setup, three-model table; hyperparameters and full tables in the appendix | Final report | Content drafted, pending paste into the final report document |

## Reminder before submission

- [ ] Decide whether to keep or remove this file before sharing the repository with the evaluators.
