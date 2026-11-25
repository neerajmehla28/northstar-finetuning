# Lab notes

A running log of what was done, what came out, and what was decided. Newest entry last. Every number here points to a saved result file or a notebook cell; none is typed in from memory.

Entry format: **date, step** | what we did | what came out | decision | next.

---

## 2026-11-25, Step 1: repo check
- **What:** read the repo, the training script and the laptop's setup.
- **Came out:** repo clean and in sync with GitHub; `train_router.py` scores the test chats on every run, has no cross-validation, reads only `customer_text`, saves only summary numbers. The laptop (7.9 GB RAM, 2-core i3, no GPU) cannot train; all compute goes to Kaggle.
- **Decision:** build on this repo (own git), share with Kawtar for integration later. Kaggle for all compute, laptop for editing only.

## 2026-11-25, dataset
- **What:** compared the notebook's `conversations_fixed.csv` with the team's `conversations.csv`.
- **Came out:** same 200 chats, splits and text; three billing-dispute escalation labels differ (v3_0027 test; v3_0110 and v3_0171 train). The team file follows the team's rule of 2026-11-11 (a question about a charge is answered, not escalated).
- **Decision:** use the team file everywhere. Copied unchanged to `data/conversations/conversations.csv` with provenance in `data/README.md` (SHA-256 checked on every load).

## 2026-11-25, Step 2: evaluation protocol (draft)
- **What:** wrote `docs/EVAL_PROTOCOL.md` and `src/northstar/finetune/protocol.py` (labels, three input variants, seeds, dataset hash check, saved folds, test lock).
- **Came out:** a random grouped split of the 160-chat pool was lopsided (one fold had 4 French chats, another 15) because only 79 customers can be placed.
- **Decision:** assign customers to folds with a balancing search (`make_folds.py`, numpy only, fixed seeds), run it once on Kaggle, commit `data/folds/folds_v1.csv` and never redraw it.
- **Next:** run `notebooks/01_protocol_and_folds.ipynb` on Kaggle, bring the folds back, then Neeraj confirms the five items at the end of the protocol.

## 2026-11-25, Step 1 result: folds
- **What:** ran `notebooks/01_protocol_and_folds.ipynb` on Kaggle; brought `folds_v1.csv`, its meta file, the balance table and chart into `data/folds/` and `results/step01/`.
- **Came out:** checked on the laptop: 160 chats, no customer split across folds in any of the three repeats, every fold has 32 chats, 13 to 14 chats that need a person, 11 to 12 French chats, and 3 to 4 chats of every request type. The three repeats are genuinely different splits (partition agreement near zero), even though they report the same objective value (the same best balance).
- **Decision:** `rep0` is the default fold set; `rep1` and `rep2` are kept for a repeated-CV check on the finalists.

## 2026-11-25, Step 2 build: CV harness and input comparison (not yet run)
- **What:** wrote `model.py` (shared encoder plus configurable heads), `train_cv.py` (cross-validated training that saves every held-out chat's raw scores at every epoch), `cv_analysis.py` and `metrics.py` (CPU-only analysis, scoring, customer-level bootstrap, exact McNemar), and `notebooks/02_cv_baseline_and_inputs.ipynb`.
- **Checked here:** all code parses; the analysis, plotting and test code ran end to end on synthetic saved runs (numbers meaningless, only the plumbing was tested); an incomplete CV is refused.
- **Not checked here:** real training (no GPU on the laptop). The notebook starts with a one-fold, two-epoch smoke test and a time estimate before the long run.
- **Bug found and fixed before the first Kaggle run:** a tiny test with a toy encoder showed that PyTorch refuses a head named `type` (`KeyError: attribute 'type' already exists`), which would have crashed the first run. Heads are now stored under `head_type`, `head_escalation`, and so on, and the plain names are restored in the model's output. The test environment was deleted afterwards; from now on all model code is run only on Kaggle.
- **Expected cost:** the original notebook took about 215 s per 30-epoch run on Kaggle; 3 inputs x 3 seeds x 5 folds = 45 runs, so the unoptimised cost would be about 2.7 hours. Per-batch padding and half precision should cut that substantially; the smoke test will measure it.
- **Next:** push, run on Kaggle with the GPU on, bring back `results_02.zip`.

## 2026-11-25, Step 2 result: cross-validated baseline and input comparison
- **What:** ran `notebooks/02_cv_baseline_and_inputs.ipynb` on a Kaggle T4 (commit `313227b`): original recipe, 3 inputs x 3 seeds x 5 folds = 45 runs, about 45 minutes (52 s, 58 s and 71 s per run for first message, customer turns and full conversation). Raw scores are in `results/cv_old_recipe/` (the 45 runs plus a config file per input); tables and figures are in `results/step02/`.
- **Came out (160 out-of-fold chats, epoch 29, mean over 3 seeds; see `results/step02/cv_summary.csv`):**
  - Type accuracy: first message 0.72, customer turns 0.70, full conversation 0.75. Averaging the 3 seeds gives 0.76, 0.74 and 0.78, so a seed ensemble is worth 3 to 5 points.
  - 95% intervals (resampling customers) are about 0.69 to 0.82, 0.68 to 0.81 and 0.70 to 0.85; every pairwise exact McNemar test has p above 0.45. **No input is measurably better than another.**
  - Seed-to-seed spread of the pooled score is only 1 to 3 points (the original notebook's 5.5-point spread was on 40 chats).
  - Escalation at the default threshold: recall about 0.75 (about 17 of the 69 chats that need a person missed), precision about 0.86, F2 about 0.77.
  - First-message input, per request type: best roadside (0.94), worst needs_human (0.56), repair_status and out_of_scope (0.69). English 0.82 vs French 0.64 (104 vs 56 chats).
- **Learning curves:** held-out macro-F1 is near zero for the first 5 epochs, climbs from epoch 7 to 17, then plateaus from about epoch 18 to 30. Held-out loss flattens near epoch 18 and does not rise; training loss keeps falling to about 0.1.
- **Decisions:**
  - Primary input by the pre-written rule: **first message** (it is 0.028 macro-F1 behind the best input, inside the 0.036 seed-to-seed spread). So the production-realistic input costs no measurable accuracy.
  - E* = 29 by the rule, but the plateau starts near epoch 18, so the exact choice inside 18 to 30 is noise. Keep 30 for the original recipe.
- **Reading and limits:** the first 15 epochs are largely spent getting started (constant learning rate 2e-5, no warmup, only 8 steps per epoch), which points to a recipe change (warmup and decay, higher learning rate) as a cheap speed-up and possibly a gain. The escalation recall and the French gap are the visible weaknesses. All differences between inputs are inside the noise at 160 chats.
- **Next:** one Kaggle notebook on the first-message input that tests (a) a language head, (b) a warmup-and-decay recipe, and (c) the escalation threshold chosen on out-of-fold scores; then gold labels for the escalation reasons.

## 2026-11-25, Track C: draft reason labels
- **What:** gave all 200 chats an escalation reason in a separate file, `data/labels/escalation_reasons.csv` (the dataset itself is untouched). Rules derived 169 rows: 113 none, 40 adversarial, 16 safety (roadside), 13 policy_limit (contested charge). I read the remaining 18 escalating chats by hand (16 needs_human, 2 warranty_claim) and drafted a reason, a confidence and a note for each.
- **Came out:** overall counts none 113, adversarial 40, safety 25, policy_limit 14, legal_threat 3, customer_request 3, bereavement 2. Half the drafted rows (9 of 18) are medium confidence, with the alternative named in the note. Rare reasons have 2 to 3 chats each, and the 40 test chats hold only 1 bereavement, 1 customer_request and no legal_threat.
- **Status:** draft, waiting for Neeraj's approval of the 18 drafted rows. Definitions follow the team's router prompt v3.
- **Decision (same day):** Neeraj parked this. The dataset has no reason column, so labels written afterwards would be invented ground truth; the tuned model is not trained or scored on reasons for now. The file stays as a parked draft (`data/labels/README.md` says so) and is not committed. Reason handling returns later via the data owner or a language-model fallback.
- **Next:** notebook 03 adds the language head and LoRA only.

## 2026-11-25, Step 3 build: language head and LoRA (not yet run)
- **What:** added LoRA to `model.py` (freeze the pretrained weights, train small add-on matrices on the attention query and value layers, rank 8, alpha 16) and `train_cv.py` (config switch, optimiser over trainable weights only, trainable-parameter count and run time saved per run); `cv_analysis.py` now scores a language head; wrote `notebooks/03_language_head_and_lora.ipynb`.
- **Models compared on the same folds, first-message input, 3 seeds:** the notebook-02 baseline (read from `results/cv_old_recipe`), the same model plus a language head, and LoRA with the language head at learning rates 1e-4 and 3e-4 (LoRA normally needs a bigger step than full fine-tuning, so two values are tried).
- **Checked here:** all code parses; the analysis cells ran end to end on synthetic runs (plumbing only). LoRA and the language head have not been run on real data; the notebook starts with a smoke test of all three models.
- **Dependency:** the notebook needs `results/cv_old_recipe` and the new code on GitHub, so push first.
- **Housekeeping:** `data/labels/` (the parked draft reason labels) is in `.gitignore` so it cannot be committed by accident.

## 2026-11-25, Kaggle setup lessons
- **What happened:** the first Kaggle run failed with `cannot import name 'protocol'`. Two causes: (1) the new files had not been pushed to GitHub yet, and (2) after the push, Kaggle still held the old clone from the first run, because the notebook only cloned when the folder did not exist.
- **Also found:** on Windows, git rewrote the dataset's line endings when staging it, which would have broken the SHA-256 check on Kaggle's Linux machine. `.gitattributes` now keeps `conversations.csv` byte-for-byte.
- **Decision:** the first cell of every Kaggle notebook deletes any old copy and clones fresh, clears stale imports, and stops with a clear message if the code is missing. Rule for later steps: push first, then run on Kaggle.
