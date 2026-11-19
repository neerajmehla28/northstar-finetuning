# northstar-finetuning

Staging repo for the fine-tuning deliverables (Member 3, Phase 5) for the MMAI 894 "Northstar Motors" project. This is **not** a standalone project — it's meant to be merged into the team repo, [`Salikawtar/northstar-agent`](https://github.com/Salikawtar/northstar-agent), as a pull request. It lives here only because this GitHub account doesn't have push access to the team repo yet.

## What's here

- `src/northstar/finetune/` — drops in at the same path in `northstar-agent`:
  - `train_router.py` — the fine-tuning pipeline (data loading, model, training loop, eval) as a reproducible CLI script.
  - `hyperparameters.json` — final settings, base-model search, training cost, stability-check summary.
  - `MODEL_CARD.md` — what the model is, why xlm-roberta-base was chosen, where the checkpoint lives (not committed — see below), and its known limitation (run-to-run accuracy instability on this small a dataset).
  - `README.md` — how to rerun it, headline result.
  - `results/` — logged CSVs: base-model comparison, learning-rate comparison, 5-run stability check, final 3-way comparison table.
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
cp -r ../northstar-finetuning/src/northstar/finetune/* src/northstar/finetune/
cp ../northstar-finetuning/docs/finetune_tasks.md docs/finetune_tasks.md

# apply the one-line settings.yaml change by hand (don't overwrite the file):
#   models.finetuned_router: "xlm-roberta-base (fine-tuned, local checkpoint; see src/northstar/finetune/)"

git add src/northstar/finetune/ docs/finetune_tasks.md config/settings.yaml
git commit -m "Add fine-tuning deliverables (Member 3, Phase 5)"
git push -u origin finetune-deliverables
# then open a PR against main on github.com
```
