"""One comparison table of every router tried, on the same 160 pool chats, from saved results only.

Run from the repo root:  python -m northstar.finetune.comparison_report
Writes results/comparison/routers_pool.csv (every router option, both inputs) and
results/comparison/training_variants.csv (every tuned-model training setup). No training, no API calls.

Same rules as notebooks 02 to 05: tuned models are scored out-of-fold at their own best epoch with the
escalation cut-off chosen on the pool by F2; the language models' repeats are combined; handoffs are scored
with the team's pipeline and grader rules (routing.py); language for tuned models comes from the word rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from northstar.finetune import baselines as B
from northstar.finetune import cv_analysis as A
from northstar.finetune import metrics as M
from northstar.finetune import protocol as P
from northstar.finetune import routing as R
from northstar.finetune import router_options as O

OUT = P.ROOT / "results" / "comparison"
VARIANTS = ("first_message", "full_conversation")
TUNED_SECONDS = 0.249        # the team's CPU measurement of the same model (MODEL_CARD.md); ours comes in the final run


def _ci(values, groups, n_boot=2000):
    v = np.asarray(values, float)
    return M.cluster_bootstrap(lambda idx: v[idx].mean(), groups, n_boot=n_boot)


def routers_table(pool: pd.DataFrame) -> pd.DataFrame:
    root = P.ROOT / "results"
    latency = pd.read_csv(root / "step04" / "latency.csv").set_index("model")["median_ms"] / 1000
    tuned_sources = {("Fine-tuned XLM-R, full", "first_message"): root / "cv_old_recipe",
                     ("Fine-tuned XLM-R, full", "full_conversation"): root / "cv_old_recipe",
                     ("Fine-tuned XLM-R, LoRA", "first_message"): root / "cv_step03b" / "lora_3e4_60ep"}
    en, fr = (pool["language"] == "en").values, (pool["language"] == "fr").values
    gold_type, gold_esc = pool["request_type"].values, pool["escalation_label"].values == 1
    groups = pool["customer_id"].values
    rows, handled = [], {}

    def add(v, family, name, ans, language_pred, llm_name=None, model_esc=None):
        s, routed = O.score(pool, ans)
        ok = ans["type"].values == gold_type
        esc = np.asarray(model_esc if model_esc is not None else ans["esc"], bool)
        hr = routed["handled_right"].values.astype(float)
        handled[(v, name)] = hr
        share = s["llm_share"]
        if llm_name is None:
            speed = f"{TUNED_SECONDS:.2f}"
        elif share >= 0.999:
            speed = f"{latency[llm_name]:.2f}"
        else:
            speed = f"{TUNED_SECONDS:.2f} + {latency[llm_name]:.2f} on {share:.0%} of chats"
        t_lo, t_hi = _ci(ok, groups)
        h_lo, h_hi = _ci(hr, groups)
        rows.append(dict(input=v, family=family, router=name,
                         type_accuracy=ok.mean(), type_ci_low=t_lo, type_ci_high=t_hi,
                         type_accuracy_en=ok[en].mean(), type_accuracy_fr=ok[fr].mean(),
                         language_accuracy=float((np.asarray(language_pred) == pool["language"].values).mean()),
                         esc_caught=int((esc & gold_esc).sum()), esc_needed=int(gold_esc.sum()),
                         esc_false_alarms=int((esc & ~gold_esc).sum()), esc_not_needed=int((~gold_esc).sum()),
                         right_team=s["right_team"], wrong_team=s["wrong_team"], missed=s["missed"],
                         attacks_right_team=s["attacks_right_team"], attacks=s["attacks"],
                         extra_handoffs=s["extra_handoffs"], handled_right=hr.mean(), handled_ci_low=h_lo, handled_ci_high=h_hi,
                         llm_share=share, usd_per_1000=s["usd_per_1000"], seconds=speed))

    sources = {m: root / "step04" / "preds" for m in ("haiku", "sonnet", "jev")}
    if (root / "step04b" / "preds").exists():                 # Jev with the same information as Haiku and Sonnet
        sources["jev2"] = root / "step04b" / "preds"
        latency = pd.concat([latency, pd.read_csv(root / "step04b" / "latency.csv").set_index("model")["median_ms"] / 1000])
    llm = {(m, v): O.combine_llm(B.load_preds(sources[m], m, v, pool)) for m in sources for v in VARIANTS}
    word_rule = {v: P.input_text(pool, v).map(R.language_rule).values for v in VARIANTS}
    names = {"haiku": "Claude Haiku 5.5 (team's router)", "sonnet": "Claude Sonnet 5.5", "jev": "Jev 1.14",
             "jev2": "Jev 1.14, same information"}
    for v in VARIANTS:
        for m in sources:
            preds = B.load_preds(sources[m], m, v, pool)
            lang = (np.mean([(p["language"] == "fr").values for p in preds], axis=0) > 0.5)
            add(v, "Language model alone", names[m], O.option_llm_alone(llm[(m, v)]), np.where(lang, "fr", "en"), m)
        for (label, tv), folder in tuned_sources.items():
            if tv != v:
                continue
            oof = A.assemble(A.load_variant(folder, v), pool)
            e_star, _ = A.pick_epoch(A.epoch_curves(oof, pool))
            cut, _ = O.choose_cutoff(oof, pool, e_star)
            t = O.tuned_answers(oof, pool, e_star, cut)
            add(v, "Fine-tuned alone", f"{label} (reason guessed from type)", O.option_her_rule(t), word_rule[v], None, t["esc"])
            for m in [k for k in ("haiku", "jev", "jev2") if k in sources]:
                add(v, "Fine-tuned + language model for the reason", f"{label} + {names[m].split(' (')[0]} reason",
                    O.option_two_step(t, llm[(m, v)]), word_rule[v], m, t["esc"])
                casc = O.option_cascade(t, llm[(m, v)])
                add(v, "Fine-tuned + language model for the reason", f"{label} + {names[m].split(' (')[0]} reason + {names[m].split(' (')[0]} when unsure",
                    casc, word_rule[v], m, casc["esc"])
    table = pd.DataFrame(rows)
    p = []
    for _, r in table.iterrows():
        ref = handled[(r["input"], names["haiku"])]
        mine = handled[(r["input"], r["router"])]
        t = M.mcnemar_exact(mine.astype(bool), ref.astype(bool))
        p.append(np.nan if r["router"] == names["haiku"] else t["p_value"])
    table["p_vs_haiku_handled"] = p
    return table


def training_table(pool: pd.DataFrame) -> pd.DataFrame:
    root = P.ROOT / "results"
    setups = [("Full fine-tuning, 2e-5, 30 epochs", root / "cv_old_recipe", "first_message", ("type", "escalation")),
              ("Full fine-tuning, 2e-5, 30 epochs", root / "cv_old_recipe", "customer_turns", ("type", "escalation")),
              ("Full fine-tuning, 2e-5, 30 epochs", root / "cv_old_recipe", "full_conversation", ("type", "escalation")),
              ("Full fine-tuning + language head", root / "cv_step03" / "full_3heads", "first_message", ("type", "escalation", "language")),
              ("LoRA 1e-4, 30 epochs (+ language head)", root / "cv_step03" / "lora_1e4", "first_message", ("type", "escalation", "language")),
              ("LoRA 3e-4, 30 epochs (+ language head)", root / "cv_step03" / "lora_3e4", "first_message", ("type", "escalation", "language")),
              ("LoRA 3e-4, 60 epochs", root / "cv_step03b" / "lora_3e4_60ep", "first_message", ("type", "escalation")),
              ("LoRA 1e-3, 60 epochs", root / "cv_step03b" / "lora_1e3_60ep", "first_message", ("type", "escalation"))]
    en, fr = (pool["language"] == "en").values, (pool["language"] == "fr").values
    rows = []
    for name, folder, v, heads in setups:
        runs = A.load_variant(folder, v)
        oof = A.assemble(runs, pool, heads)
        e_star, _ = A.pick_epoch(A.epoch_curves(oof, pool))
        s = A.summary_at_epoch(oof, pool, e_star)
        ps = s["per_seed"]
        ens_ok = s["ensemble_type_logits"].argmax(-1) == pool["type_label"].values
        rows.append(dict(setup=name, input=v, epoch=e_star,
                         type_accuracy_mean=ps["type_accuracy"].mean(), type_accuracy_sd=ps["type_accuracy"].std(),
                         type_macro_f1_mean=ps["type_macro_f1"].mean(), combined_type_accuracy=ens_ok.mean(),
                         combined_type_accuracy_en=ens_ok[en].mean(), combined_type_accuracy_fr=ens_ok[fr].mean(),
                         esc_recall_default=ps["esc_recall"].mean(), esc_precision_default=ps["esc_precision"].mean(),
                         language_head_accuracy=ps["language_accuracy"].mean() if "language_accuracy" in ps else np.nan,
                         trained_numbers=int(np.mean([r["n_trainable"] for r in runs])) if "n_trainable" in runs[0] else 278052107,
                         seconds_per_run=float(np.mean([r["seconds"] for r in runs])) if "seconds" in runs[0] else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    pool = P.pool(P.load_all())
    OUT.mkdir(parents=True, exist_ok=True)
    routers = routers_table(pool)
    routers.to_csv(OUT / "routers_pool.csv", index=False)
    training = training_table(pool)
    training.to_csv(OUT / "training_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(routers[["input", "router", "type_accuracy", "type_accuracy_fr", "esc_caught", "esc_false_alarms", "right_team",
                   "attacks_right_team", "extra_handoffs", "handled_right", "p_vs_haiku_handled", "usd_per_1000"]].round(3).to_string())
    print(training.round(3).to_string())


if __name__ == "__main__":
    main()
