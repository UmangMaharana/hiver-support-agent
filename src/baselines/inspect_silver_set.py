from pathlib import Path

import pandas as pd


SILVER_PATH = Path("data/processed/tesco_silver_set.parquet")

SAMPLES_PER_INTENT = 5
RANDOM_SEED = 42


def main():
    df = pd.read_parquet(SILVER_PATH)

    print(f"Silver examples: {len(df):,}")
    print(f"Intents: {df['silver_intent'].nunique()}")

    print("\n" + "=" * 100)
    print("RANDOM SILVER-SET SANITY SAMPLE")
    print("=" * 100)

    for intent in sorted(df["silver_intent"].unique()):
        subset = df[df["silver_intent"] == intent]

        sample = subset.sample(
            n=min(SAMPLES_PER_INTENT, len(subset)),
            random_state=RANDOM_SEED,
        )

        print("\n" + "-" * 100)
        print(f"INTENT: {intent} | available={len(subset)}")
        print("-" * 100)

        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            print(f"\n[{i}]")
            print(f"Message: {row['initial_customer_message']}")
            print(f"Matched rule: {row['matched_rule']}")


if __name__ == "__main__":
    main()