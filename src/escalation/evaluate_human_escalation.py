from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"
OUTPUT = ROOT / "data" / "escalation" / "human_escalation_metrics.csv"


def parse_bool(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    return None


def evaluate(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "agreement": sum(
            a == b for a, b in zip(y_true, y_pred)
        ) / len(y_true),
    }


def print_confusion(y_true, y_pred, name):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )

    print(f"\n{name} confusion matrix")
    print("                 Pred NO   Pred YES")
    print(f"Actual NO        {cm[0, 0]:8d}   {cm[0, 1]:8d}")
    print(f"Actual YES       {cm[1, 0]:8d}   {cm[1, 1]:8d}")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "golden_id",
        "gold_intent",
        "escalate",
        "policy_escalate",
        "human_escalate",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Parse boolean columns.
    for col in [
        "escalate",
        "policy_escalate",
        "human_escalate",
    ]:
        df[col] = df[col].apply(parse_bool)

    # Keep only rows with valid human labels.
    eval_df = df[df["human_escalate"].notna()].copy()

    if eval_df.empty:
        raise ValueError("No valid human annotations found.")

    y_true = eval_df["human_escalate"].astype(bool).tolist()
    y_gemini = eval_df["escalate"].astype(bool).tolist()
    y_rule = eval_df["policy_escalate"].astype(bool).tolist()

    print("=" * 72)
    print("HUMAN ESCALATION EVALUATION")
    print("=" * 72)

    print(f"Examples evaluated: {len(eval_df)}")
    print(f"Human YES: {sum(y_true)}")
    print(f"Human NO:  {len(y_true) - sum(y_true)}")
    print(f"Human escalation rate: {sum(y_true) / len(y_true):.1%}")

    # Overall metrics.
    results = [
        evaluate(y_true, y_gemini, "Gemini"),
        evaluate(y_true, y_rule, "Rule baseline"),
    ]

    for result in results:
        print(f"\n--- {result['model']} vs Human ---")
        print(f"Accuracy:   {result['accuracy']:.3f}")
        print(f"Precision:  {result['precision']:.3f}")
        print(f"Recall:     {result['recall']:.3f}")
        print(f"F1:         {result['f1']:.3f}")
        print(f"Agreement:  {result['agreement']:.3f}")

    # Confusion matrices.
    print_confusion(
        y_true,
        y_gemini,
        "Gemini",
    )

    print_confusion(
        y_true,
        y_rule,
        "Rule baseline",
    )

    # Per-intent breakdown.
    print("\n" + "=" * 72)
    print("PER-INTENT BREAKDOWN")
    print("=" * 72)

    rows = []

    for intent, group in eval_df.groupby("gold_intent"):
        true = group["human_escalate"].astype(bool).tolist()
        gemini = group["escalate"].astype(bool).tolist()
        rule = group["policy_escalate"].astype(bool).tolist()

        rows.append(
            {
                "intent": intent,
                "n": len(group),
                "human_yes": sum(true),
                "human_rate": sum(true) / len(true),
                "gemini_rate": sum(gemini) / len(gemini),
                "rule_rate": sum(rule) / len(rule),
                "gemini_agreement": sum(
                    a == b for a, b in zip(true, gemini)
                ) / len(true),
                "rule_agreement": sum(
                    a == b for a, b in zip(true, rule)
                ) / len(true),
            }
        )

    intent_df = pd.DataFrame(rows).sort_values("intent")

    print(intent_df.to_string(index=False))

    # Save headline metrics.
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUTPUT, index=False)

    print(f"\nSaved metrics: {OUTPUT}")


if __name__ == "__main__":
    main()