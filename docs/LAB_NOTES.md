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

## 2026-11-25, Kaggle setup lessons
- **What happened:** the first Kaggle run failed with `cannot import name 'protocol'`. Two causes: (1) the new files had not been pushed to GitHub yet, and (2) after the push, Kaggle still held the old clone from the first run, because the notebook only cloned when the folder did not exist.
- **Also found:** on Windows, git rewrote the dataset's line endings when staging it, which would have broken the SHA-256 check on Kaggle's Linux machine. `.gitattributes` now keeps `conversations.csv` byte-for-byte.
- **Decision:** the first cell of every Kaggle notebook deletes any old copy and clones fresh, clears stale imports, and stops with a clear message if the code is missing. Rule for later steps: push first, then run on Kaggle.
