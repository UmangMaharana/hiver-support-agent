from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.escalation.escalation_policy import should_escalate


ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
GENERATION_PATH = ROOT / "data" / "generation" / "reply_predictions.csv"
OUTPUT_PATH = ROOT / "data" / "escalation" / "escalation_policy_predictions.csv"


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def main():
    print("=" * 80)
    print("ESCALATION POLICY EVALUATION")
    print("=" * 80)

    golden = pd.read_csv(GOLDEN_PATH)
    generation = pd.read_csv(GENERATION_PATH)

    df = golden[
        [
            "golden_id",
            "intent",
            "initial_customer_message",
        ]
    ].rename(
        columns={
            "intent": "gold_intent",
            "initial_customer_message": "customer_message",
        }
    )

    generation_subset = generation[
        [
            "golden_id",
            "predicted_intent",
            "escalate",
            "escalation_reason",
        ]
    ]

    df = df.merge(
        generation_subset,
        on="golden_id",
        how="left",
    )

    policy_results = []

    for _, row in df.iterrows():
        predicted_intent = row["predicted_intent"]

        policy_escalate, policy_reason = should_escalate(
            row["customer_message"],
            predicted_intent,
        )

        policy_results.append(
            {
                "golden_id": row["golden_id"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": predicted_intent,
                "customer_message": row["customer_message"],
                "gemini_escalate": normalize_bool(row["escalate"]),
                "gemini_reason": row["escalation_reason"],
                "policy_escalate": policy_escalate,
                "policy_reason": policy_reason,
            }
        )

    results = pd.DataFrame(policy_results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    gemini = results["gemini_escalate"]
    policy = results["policy_escalate"]

    print()
    print(f"Golden examples: {len(results)}")

    print()
    print("ESCALATION RATES")
    print("-" * 40)
    print(f"Gemini: {gemini.mean():.1%} ({gemini.sum()}/{len(gemini)})")
    print(f"Policy: {policy.mean():.1%} ({policy.sum()}/{len(policy)})")

    # There is no human escalation ground truth yet.
    # These metrics therefore compare the deterministic policy
    # against Gemini rather than claiming either is correct.
    print()
    print("POLICY vs GEMINI AGREEMENT")
    print("-" * 40)

    agreement = (gemini == policy).mean()

    print(f"Agreement: {agreement:.1%}")
    print(
        f"Disagreements: {(gemini != policy).sum()}/{len(results)}"
    )

    print()
    print("Treating Gemini as a provisional reference only:")
    print(f"Accuracy:  {accuracy_score(gemini, policy):.3f}")
    print(
        f"Precision: {precision_score(gemini, policy, zero_division=0):.3f}"
    )
    print(
        f"Recall:    {recall_score(gemini, policy, zero_division=0):.3f}"
    )
    print(
        f"F1:        {f1_score(gemini, policy, zero_division=0):.3f}"
    )

    print()
    print("CONFUSION")
    print("-" * 40)

    confusion = pd.crosstab(
        gemini,
        policy,
        rownames=["Gemini"],
        colnames=["Policy"],
    )

    print(confusion)

    print()
    print("REPRESENTATIVE DISAGREEMENTS")
    print("-" * 40)

    disagreements = results[gemini != policy]

    for _, row in disagreements.head(20).iterrows():
        print()
        print(row["golden_id"])
        print(f"Gold intent:      {row['gold_intent']}")
        print(f"Predicted intent: {row['predicted_intent']}")
        print(f"Customer:         {row['customer_message']}")
        print(f"Gemini:           {row['gemini_escalate']}")
        print(f"Policy:           {row['policy_escalate']}")
        print(f"Policy reason:    {row['policy_reason']}")

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()