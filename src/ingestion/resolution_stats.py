from pathlib import Path
import sqlite3
import re

import pandas as pd


DB_PATH = Path("data/processed/twcs.db")

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
]

CHUNK_SIZE = 100_000

# Interpretable support/action signals.
RESOLUTION_PATTERNS = {
    "refund": r"\brefund(ed|s)?\b",
    "replace": r"\breplac(e|ed|ement|ing)\b",
    "investigate": r"\binvestigat(e|ed|ion|ing)\b",
    "contact": r"\bcontact\b",
    "dm": r"\bDM\b|\bdirect message\b",
    "email": r"\bemail\b",
    "complaint": r"\bcomplaint\b",
    "management": r"\bmanagement\b",
    "store": r"\bstore\b",
    "provide": r"\bprovide\b",
    "send": r"\bsend\b",
    "check": r"\bcheck\b",
    "team": r"\bteam\b",
}


def main() -> None:
    connection = sqlite3.connect(DB_PATH)

    try:
        placeholders = ",".join("?" for _ in BRANDS)

        query = f"""
            SELECT
                author_id,
                inbound,
                text
            FROM tweets
            WHERE author_id IN ({placeholders})
        """

        df_iter = pd.read_sql_query(
            query,
            connection,
            params=BRANDS,
            chunksize=CHUNK_SIZE,
        )

        counts = {
            brand: {
                "tweets": 0,
                "action_tweets": 0,
                "signals": {name: 0 for name in RESOLUTION_PATTERNS},
            }
            for brand in BRANDS
        }

        for df in df_iter:
            for brand in BRANDS:
                brand_df = df[df["author_id"] == brand]

                if brand_df.empty:
                    continue

                counts[brand]["tweets"] += len(brand_df)

                texts = (
                    brand_df["text"]
                    .fillna("")
                    .astype(str)
                )

                for text in texts:
                    matched = False

                    for name, pattern in RESOLUTION_PATTERNS.items():
                        if re.search(pattern, text, flags=re.IGNORECASE):
                            counts[brand]["signals"][name] += 1
                            matched = True

                    if matched:
                        counts[brand]["action_tweets"] += 1

    finally:
        connection.close()

    print("=" * 90)
    print("SUPPORT ACTION / RESOLUTION SIGNAL ANALYSIS")
    print("=" * 90)

    print(
        f"\n{'Brand':<18}"
        f"{'Tweets':>12}"
        f"{'Action':>12}"
        f"{'Action %':>12}"
    )

    print("-" * 60)

    for brand in sorted(
        BRANDS,
        key=lambda x: counts[x]["action_tweets"],
        reverse=True,
    ):
        c = counts[brand]

        action_pct = (
            c["action_tweets"] / c["tweets"] * 100
            if c["tweets"]
            else 0
        )

        print(
            f"{brand:<18}"
            f"{c['tweets']:>12,}"
            f"{c['action_tweets']:>12,}"
            f"{action_pct:>11.1f}%"
        )

    print("\nSignal breakdown:")
    print("-" * 90)

    for brand in BRANDS:
        print(f"\n{brand}")

        for signal, count in sorted(
            counts[brand]["signals"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {signal:<15} {count:>10,}")


if __name__ == "__main__":
    main()