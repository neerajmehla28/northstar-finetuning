# northstar-finetuning

Member 3 (fine-tuning and model comparison) of the MMAI 894 "Northstar Motors" project: a fine-tuned router compared with prompted Claude Haiku and Sonnet and with Jev, on the team's dataset, with the team's own handoff and grading rules. The team's agent lives in [`Salikawtar/northstar-agent`](https://github.com/Salikawtar/northstar-agent).

## How to check this work (for a reviewer)

No secrets are stored anywhere in this repo. Nothing below needs an API key or a GPU unless stated.

1. **Tests (a few seconds):** `pip install -r requirements.txt`, then `pytest` from the repo root. Besides the rules and parsers, `tests/test_reported_numbers.py` recomputes headline numbers from the saved runs and answers and checks they match what is reported.
2. **The comparison tables:** `PYTHONPATH=src python -m northstar.finetune.comparison_report` rebuilds `results/comparison/` from the saved results.
3. **The notebooks** (`notebooks/`, numbered in run order; each one's executed copy with all outputs is in `results/stepNN/`):
   - 01, 05: CPU only.
   - 02, 03, 03b, 03c, 06: training, need a GPU (they were run on a free Kaggle T4).
   - 04, 04b, 06, 07: call language models through OpenRouter. **Without a key they replay the answers saved in the repo, for free**, and stop with a clear message at any call that was not saved. With a key (a Kaggle secret named `OPENROUTER_API_KEY`), saved answers are still reused and never paid for twice.
4. **What was decided and why:** `docs/EVAL_PROTOCOL.md` (the rules, written before the results they govern) and `docs/LAB_NOTES.md` (every step, result and decision, each number pointing to a saved file).
5. **The test chats** are locked in code (`protocol.test_set`, `protocol.fresh_set`) and were opened once, in notebook 06, with settings frozen beforehand in `config/final_settings.json`; each opening is logged.

---

## Earlier notes (first version of this repo, kept for history)

Staging repo for the fine-tuning deliverables (Member 3, Phase 5) for the MMAI 894 "Northstar Motors" project. This is **not** a standalone project — it's meant to be merged into the team repo, [`Salikawtar/northstar-agent`](https://github.com/Salikawtar/northstar-agent), as a pull request. It lives here only because this GitHub account doesn't have push access to the team repo yet.

## What's here

- `src/northstar/finetune/` — drops in at the same path in `northstar-agent`:
  - `train_router.py` — the fine-tuning pipeline (data loading, model, training loop, eval) as a reproducible CLI script.
  - `hyperparameters.json` — final settings, base-model search, training cost, stability-check summary.
  - `MODEL_CARD.md` — what the model is, why xlm-roberta-base was chosen, where the checkpoint lives (not committed — see below), and its known limitation (run-to-run accuracy instability on this small a dataset).
  - `README.md` — how to rerun it, headline result.
  - `results/` — logged CSVs: base-model comparison, learning-rate comparison, 5-run stability check, final 3-way comparison table.
  - `dealer-chatbot-router-model-tuning.ipynb` — the final Kaggle notebook this work was developed in, run end-to-end with 0 errors. Source of truth for the exact run that produced every number above; `train_router.py` is the same pipeline cleaned up into a standalone script.
- `config/settings.yaml.snippet` — the team's full `config/settings.yaml` with one line filled in (`models.finetuned_router`); **not** meant to overwrite the real file — apply just that one change when merging (see below).
- `docs/finetune_tasks.md` — the team's own task tracker for this phase, with rows 2-14 updated to Done.

**Not included:** the trained checkpoint (`xlmr_router_best.pt`, ~1.1GB) — too large for a normal git repo. `MODEL_CARD.md` explains how to reproduce it in about 3.5 minutes for under $0.04.

## How to merge into the team repo

Once you have push access (or ask a teammate who does) on `Salikawtar/northstar-agent`:

```bash
git clone https://github.com/Salikawtar/northstar-agent
cd northstar-agent
git checkout -b finetune-deliverables

# copy the matching files in from this repo
cp -r ../northstar-finetuning/src/northstar/finetune/. src/northstar/finetune/
cp ../northstar-finetuning/docs/finetune_tasks.md docs/finetune_tasks.md

# apply the one-line settings.yaml change by hand (don't overwrite the file):
#   models.finetuned_router: "xlm-roberta-base (fine-tuned, local checkpoint; see src/northstar/finetune/)"

git add src/northstar/finetune/ docs/finetune_tasks.md config/settings.yaml
git commit -m "Add fine-tuning deliverables (Member 3, Phase 5)"
git push -u origin finetune-deliverables
# then open a PR against main on github.com
```
