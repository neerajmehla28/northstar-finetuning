"""Fine-tuning script for the router (Phase 5, Member 3: Neeraj).

Trains a small encoder with two linear heads directly on top of the [CLS]
token — one head for the 10-way request type, one for the binary escalation
decision — on the 130 training chats, picks the checkpoint with the lowest
validation loss, and evaluates once on the 40 held-out test chats.

This is a cleaned-up, standalone version of the pipeline developed and run
interactively in the Kaggle notebook (free T4 GPU). It reproduces the exact
same data loading, model, and training loop as the notebook; it does not
reproduce the notebook's cell-by-cell narrative (learning-rate sweep,
candidate-model comparison, stability check) — those are one-off analyses,
run here as repeated calls to `train_and_evaluate`, not separate scripts.

Why this task, why this split:
    The router is the one component every chat passes through, so its
    speed and cost matter more than any other piece of the agent. It is
    also a bounded, well-defined classification task (10 request types +
    1 binary flag), which is exactly the shape fine-tuning handles well on
    small data (as opposed to open-ended generation). Train/validation/test
    is 130/30/40 chats, grouped by customer so no customer appears in more
    than one split, test touched exactly once at the end.

Usage:
    # Single run, final settings (what produced xlmr_router_best.pt)
    python -m northstar.finetune.train_router --model xlm-roberta-base --lr 2e-5 --epochs 30 --save best_model.pt

    # Learning-rate sweep (reproduces Step 14 of the notebook)
    python -m northstar.finetune.train_router --model xlm-roberta-base --lr-sweep 1e-5 2e-5 5e-5

    # Stability check: same settings, repeated N times, logged to CSV (Step 23)
    python -m northstar.finetune.train_router --model xlm-roberta-base --lr 2e-5 --stability-runs 5

Data path defaults to the repo's own data/conversations/conversations.csv
(see config/settings.yaml -> paths.conversations). On Kaggle this was
/kaggle/input/datasets/<user>/chats-data/conversations_fixed.csv; pass
--data to point at either.

No secrets are used or required by this script. Training runs entirely on
local file data and a local or free-tier GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer

REQUEST_TYPES = [
    "book_service", "repair_status", "warranty_claim", "recall_inquiry",
    "roadside_assistance", "parts_order", "billing_dispute", "lease_end_return",
    "out_of_scope", "needs_human",
]
LABEL2ID = {label: i for i, label in enumerate(REQUEST_TYPES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "conversations" / "conversations.csv"


class RouterClassifier(nn.Module):
    """Shared encoder + two linear heads: request type (10-way) and escalation (binary)."""

    def __init__(self, model_name: str, num_types: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden_size, num_types)
        self.escalation_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = self.dropout(outputs.last_hidden_state[:, 0, :])  # [CLS] token
        return self.type_head(cls_vector), self.escalation_head(cls_vector).squeeze(-1)


def load_splits(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df["type_label"] = df["request_type"].map(LABEL2ID)
    df["escalation_label"] = df["needs_escalation"].astype(int)
    assert df["type_label"].isna().sum() == 0, "Found a request_type not in REQUEST_TYPES"
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def train_and_evaluate(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    epochs: int = 30,
    max_length: int = 256,
    batch_size: int = 16,
    lr: float = 2e-5,
    verbose: bool = True,
) -> tuple[dict, list[dict], dict]:
    """Trains one model end to end and evaluates once on the test split.

    Returns (result_dict, per_epoch_history, best_state_dict) so a caller
    can save the winning checkpoint without retraining.
    """
    tok = AutoTokenizer.from_pretrained(model_name)

    def enc(texts):
        return tok(list(texts), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    tr_enc, va_enc, te_enc = enc(train_df["customer_text"]), enc(val_df["customer_text"]), enc(test_df["customer_text"])

    tr_type = torch.tensor(train_df["type_label"].values, dtype=torch.long)
    tr_esc = torch.tensor(train_df["escalation_label"].values, dtype=torch.float)
    va_type = torch.tensor(val_df["type_label"].values, dtype=torch.long)
    va_esc = torch.tensor(val_df["escalation_label"].values, dtype=torch.float)

    tr_loader = DataLoader(TensorDataset(tr_enc["input_ids"], tr_enc["attention_mask"], tr_type, tr_esc),
                            batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_enc["input_ids"], va_enc["attention_mask"], va_type, va_esc),
                            batch_size=batch_size, shuffle=False)

    m = RouterClassifier(model_name, num_types=len(REQUEST_TYPES)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    type_loss_fn = nn.CrossEntropyLoss()
    escalation_loss_fn = nn.BCEWithLogitsLoss()

    def run_epoch(loader, training: bool) -> float:
        m.train() if training else m.eval()
        total_loss, n = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for ids, mask, ty, es in loader:
                ids, mask, ty, es = ids.to(device), mask.to(device), ty.to(device), es.to(device)
                ty_logits, es_logits = m(ids, mask)
                loss = type_loss_fn(ty_logits, ty) + escalation_loss_fn(es_logits, es)
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * ids.size(0)
                n += ids.size(0)
        return total_loss / n

    best_val, best_state = float("inf"), None
    hist = []
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(tr_loader, True)
        va_loss = run_epoch(va_loader, False)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best_val:
            best_val, best_state = va_loss, {k: v.clone() for k, v in m.state_dict().items()}
        if verbose and ep % 5 == 0:
            print(f"  epoch {ep:2d}  train_loss={tr_loss:.3f}  val_loss={va_loss:.3f}")

    m.load_state_dict(best_state)
    m.eval()

    latencies_ms = []
    with torch.no_grad():
        ty_logits, es_logits = m(te_enc["input_ids"].to(device), te_enc["attention_mask"].to(device))
        preds_type = ty_logits.argmax(dim=1).cpu().numpy()
        preds_esc = (es_logits > 0).long().cpu().numpy()
        for i in range(len(test_df)):
            ids = te_enc["input_ids"][i:i + 1].to(device)
            mask = te_enc["attention_mask"][i:i + 1].to(device)
            start = time.perf_counter()
            _ = m(ids, mask)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    true_type = test_df["type_label"].values
    true_esc = test_df["escalation_label"].values
    by_lang = pd.DataFrame({
        "correct": preds_type == true_type,
        "language": test_df["language"].values,
    }).groupby("language")["correct"].mean()

    result = {
        "model": model_name,
        "n_params": sum(p.numel() for p in m.parameters()),
        "lr": lr,
        "type_accuracy": accuracy_score(true_type, preds_type),
        "macro_f1": f1_score(true_type, preds_type, average="macro"),
        "escalation_accuracy": accuracy_score(true_esc, preds_esc),
        "escalation_recall": recall_score(true_esc, preds_esc),
        "escalation_precision": precision_score(true_esc, preds_esc),
        "en_type_accuracy": by_lang.get("en"),
        "fr_type_accuracy": by_lang.get("fr"),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_p90": float(np.percentile(latencies_ms, 90)),
        "cost_usd_per_chat": 0.0,
        "best_epoch": hist[int(np.argmin([h["val_loss"] for h in hist]))]["epoch"],
    }
    return result, hist, best_state


def _free_gpu(*names, local_vars):
    """Drop references and clear CUDA's caching allocator between runs.

    Sequential training runs in one process otherwise accumulate GPU memory
    PyTorch doesn't release on its own, which reliably OOMs the next run.
    """
    for name in names:
        local_vars.pop(name, None)
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the conversations CSV")
    p.add_argument("--model", default="xlm-roberta-base", help="HF model name for the shared encoder")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=Path, default=None, help="Where to save the best checkpoint (.pt)")
    p.add_argument("--results", type=Path, default=Path(__file__).parent / "results",
                    help="Directory to write result CSVs into")
    p.add_argument("--lr-sweep", type=float, nargs="+", default=None,
                    help="Run once per learning rate given, log all to one CSV, skip the single --lr run")
    p.add_argument("--stability-runs", type=int, default=None,
                    help="Repeat training N times with identical settings, log each run to one CSV")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df, val_df, test_df = load_splits(args.data)
    print(f"Train: {len(train_df)}  Validation: {len(val_df)}  Test: {len(test_df)}")

    args.results.mkdir(parents=True, exist_ok=True)

    if args.lr_sweep:
        rows = []
        for lr in args.lr_sweep:
            start = time.time()
            result, _, _ = train_and_evaluate(args.model, train_df, val_df, test_df, device,
                                                epochs=args.epochs, max_length=args.max_length,
                                                batch_size=args.batch_size, lr=lr, verbose=not args.quiet)
            result["train_time_sec"] = time.time() - start
            rows.append(result)
            print(f"lr={lr}: type_accuracy={result['type_accuracy']:.3f}  macro_f1={result['macro_f1']:.3f}")
            gc.collect()
            torch.cuda.empty_cache()
        out = args.results / "learning_rate_comparison.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print("Saved", out)
        return

    if args.stability_runs:
        rows = []
        for run_id in range(args.stability_runs):
            start = time.time()
            result, _, _ = train_and_evaluate(args.model, train_df, val_df, test_df, device,
                                                epochs=args.epochs, max_length=args.max_length,
                                                batch_size=args.batch_size, lr=args.lr, verbose=not args.quiet)
            result["run_id"] = run_id
            result["train_time_sec"] = time.time() - start
            rows.append(result)
            print(f"Run {run_id}: type_accuracy={result['type_accuracy']:.3f}  macro_f1={result['macro_f1']:.3f}")
            gc.collect()
            torch.cuda.empty_cache()
        stability_df = pd.DataFrame(rows)
        out = args.results / "stability_runs.csv"
        stability_df.to_csv(out, index=False)
        print()
        print(f"Mean type_accuracy: {stability_df['type_accuracy'].mean():.3f}  "
              f"(std: {stability_df['type_accuracy'].std():.3f})")
        print(f"Range: {stability_df['type_accuracy'].min():.3f} to {stability_df['type_accuracy'].max():.3f}")
        print("Saved", out)
        return

    start = time.time()
    result, hist, best_state = train_and_evaluate(args.model, train_df, val_df, test_df, device,
                                                    epochs=args.epochs, max_length=args.max_length,
                                                    batch_size=args.batch_size, lr=args.lr, verbose=not args.quiet)
    result["train_time_sec"] = time.time() - start
    print(json.dumps(result, indent=2, default=float))

    if args.save:
        torch.save(best_state, args.save)
        print("Saved checkpoint to", args.save)

    out = args.results / "final_run.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print("Saved", out)


if __name__ == "__main__":
    main()
