from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"
OUTPUT_PATH = INPUT_PATH

BATCH_SIZE = 10


def is_completed(group):
    value = group["human_escalate"]

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "false",
        "yes",
        "no",
        "1",
        "0",
    }


def normalize_decision(value):
    value = value.strip().lower()

    if value in {"y", "yes", "true", "1"}:
        return "True"

    if value in {"n", "no", "false", "0"}:
        return "False"

    return None


def main():
    print("=" * 80)
    print("HUMAN ESCALATION ANNOTATION")
    print("=" * 80)

    df = pd.read_csv(INPUT_PATH)

    # Make sure annotation columns exist.
    if "human_escalate" not in df.columns:
        df["human_escalate"] = ""

    if "human_reason" not in df.columns:
        df["human_reason"] = ""

    while True:
        pending = []

        for index, row in df.iterrows():
            if not is_completed(row):
                pending.append(index)

        if not pending:
            print("\nAll 50 examples are already annotated.")
            break

        batch = pending[:BATCH_SIZE]

        print()
        print(
            f"Pending: {len(pending)} | "
            f"Showing {len(batch)} examples"
        )
        print("=" * 80)

        for index in batch:
            row = df.loc[index]

            print()
            print(f"{row['golden_id']}")
            print("-" * 80)
            print(f"Gold intent:      {row['gold_intent']}")
            print(f"Predicted intent: {row['predicted_intent']}")
            print()
            print(f"Customer:")
            print(row["customer_message"])
            print()
            print(f"Gemini escalate:  {row['escalate']}")
            print(f"Gemini reason:    {row['escalation_reason']}")
            print()
            print(f"Rule escalate:    {row['policy_escalate']}")
            print(f"Rule reason:      {row['policy_reason']}")
            print()

            while True:
                decision = input(
                    "Human escalation? [y/n, q=quit]: "
                ).strip().lower()

                if decision == "q":
                    df.to_csv(
                        OUTPUT_PATH,
                        index=False,
                    )

                    print()
                    print("Progress saved.")
                    print(f"Saved: {OUTPUT_PATH}")
                    return

                normalized = normalize_decision(decision)

                if normalized is not None:
                    break

                print("Please enter y, n, or q.")

            reason = input(
                "Reason (short): "
            ).strip()

            df.at[index, "human_escalate"] = normalized
            df.at[index, "human_reason"] = reason

            # Save after every annotation so progress is never lost.
            df.to_csv(
                OUTPUT_PATH,
                index=False,
            )

        print()
        print(
            f"Batch complete. "
            f"{len(pending) - len(batch)} remaining."
        )

    print()
    print("=" * 80)
    print("ANNOTATION COMPLETE")
    print("=" * 80)

    completed = df["human_escalate"].apply(
        lambda x: str(x).strip().lower() in {
            "true",
            "false",
        }
    )

    print(f"Completed: {completed.sum()}/{len(df)}")
    print()
    print("Human escalation distribution:")
    print(
        df.loc[completed, "human_escalate"]
        .value_counts()
        .to_string()
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()