from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
GENERATION_PATH = ROOT / "data" / "generation" / "reply_predictions.csv"
POLICY_PATH = ROOT / "data" / "escalation" / "escalation_policy_predictions.csv"

OUTPUT_PATH = ROOT / "data" / "escalation" / "escalation_human_eval_50.csv"

TARGET_SIZE = 50
RANDOM_STATE = 42


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
    print("BUILDING HUMAN ESCALATION EVALUATION SET")
    print("=" * 80)

    golden = pd.read_csv(GOLDEN_PATH)

    generation = pd.read_csv(
        GENERATION_PATH,
        usecols=[
            "golden_id",
            "predicted_intent",
            "escalate",
            "escalation_reason",
        ],
    )

    policy = pd.read_csv(
        POLICY_PATH,
        usecols=[
            "golden_id",
            "policy_escalate",
            "policy_reason",
        ],
    )

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

    df = df.merge(
        generation,
        on="golden_id",
        how="inner",
    )

    df = df.merge(
        policy,
        on="golden_id",
        how="inner",
    )

    if len(df) < TARGET_SIZE:
        raise ValueError(
            f"Only {len(df)} usable examples available; "
            f"cannot sample {TARGET_SIZE}."
        )

    # Stratified sampling by gold intent.
    # This ensures the human benchmark isn't dominated by "other"
    # and product-quality examples.
    selected_parts = []

    intents = sorted(df["gold_intent"].unique())

    per_intent = TARGET_SIZE // len(intents)
    remainder = TARGET_SIZE % len(intents)

    for i, intent in enumerate(intents):
        group = df[df["gold_intent"] == intent]

        n = per_intent + (1 if i < remainder else 0)
        n = min(n, len(group))

        selected_parts.append(
            group.sample(
                n=n,
                random_state=RANDOM_STATE + i,
            )
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # If some small intents prevented us from reaching 50,
    # fill the remaining slots from unused examples.
    if len(selected) < TARGET_SIZE:
        remaining = df[
            ~df["golden_id"].isin(selected["golden_id"])
        ]

        extra = remaining.sample(
            TARGET_SIZE - len(selected),
            random_state=RANDOM_STATE,
        )

        selected = pd.concat(
            [selected, extra],
            ignore_index=True,
        )

    # Shuffle final benchmark.
    selected = selected.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    selected["human_escalate"] = ""
    selected["human_reason"] = ""

    selected = selected[
        [
            "golden_id",
            "gold_intent",
            "customer_message",
            "predicted_intent",
            "escalate",
            "escalation_reason",
            "policy_escalate",
            "policy_reason",
            "human_escalate",
            "human_reason",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Total golden examples: {len(df)}")
    print(f"Human evaluation examples: {len(selected)}")

    print()
    print("INTENT DISTRIBUTION")
    print("-" * 40)
    print(
        selected["gold_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("GEMINI ESCALATION RATE")
    print("-" * 40)
    print(
        f"{selected['escalate'].map(normalize_bool).mean():.1%}"
    )

    print()
    print("POLICY ESCALATION RATE")
    print("-" * 40)
    print(
        f"{selected['policy_escalate'].map(normalize_bool).mean():.1%}"
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()