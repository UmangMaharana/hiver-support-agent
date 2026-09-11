from pathlib import Path
import re
import sqlite3

import pandas as pd


DB_PATH = Path("data/processed/twcs.db")
OUTPUT_PATH = Path("data/processed/tesco_customer_messages.parquet")

BRAND = "Tesco"


def parse_response_ids(value: str) -> list[int]:
    """Parse comma-separated response tweet IDs."""
    if not value:
        return []

    ids = []

    for item in str(value).split(","):
        item = item.strip()

        if not item:
            continue

        try:
            ids.append(int(float(item)))
        except ValueError:
            continue

    return ids


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(DB_PATH)

    try:
        print("Finding Tesco's outbound tweets...")

        tesco = pd.read_sql_query(
            """
            SELECT
                tweet_id,
                response_tweet_id
            FROM tweets
            WHERE author_id = ?
              AND inbound = 0
              AND response_tweet_id IS NOT NULL
            """,
            connection,
            params=(BRAND,),
        )

        print(
            f"Tesco outbound tweets with responses: "
            f"{len(tesco):,}"
        )

        customer_ids = set()

        for value in tesco["response_tweet_id"]:
            customer_ids.update(
                parse_response_ids(value)
            )

        print(
            f"Unique referenced tweet IDs: "
            f"{len(customer_ids):,}"
        )

        # SQLite has a parameter limit, so query in batches.
        customer_ids = list(customer_ids)

        rows = []

        batch_size = 900

        for start in range(
            0,
            len(customer_ids),
            batch_size,
        ):
            batch = customer_ids[
                start : start + batch_size
            ]

            placeholders = ",".join(
                "?" for _ in batch
            )

            query = f"""
                SELECT
                    tweet_id,
                    author_id,
                    inbound,
                    created_at,
                    text,
                    in_response_to_tweet_id
                FROM tweets
                WHERE tweet_id IN ({placeholders})
                  AND inbound = 1
            """

            result = connection.execute(
                query,
                batch,
            ).fetchall()

            rows.extend(result)

        df = pd.DataFrame(
            rows,
            columns=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "in_response_to_tweet_id",
            ],
        )

    finally:
        connection.close()

    if df.empty:
        print("No customer messages found.")
        return

    # Basic text cleanup.
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"https?://\S+",
            " ",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    # Remove obvious empty messages.
    df = df[df["text"].str.len() > 0].copy()

    # Remove exact duplicate tweet IDs.
    df = df.drop_duplicates(
        subset=["tweet_id"]
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("TESCO CUSTOMER MESSAGE EXTRACTION")
    print("=" * 80)

    print(f"\nMessages extracted: {len(df):,}")
    print(f"Unique customers:   {df['author_id'].nunique():,}")
    print(f"Output:             {OUTPUT_PATH}")

    print("\nSample messages:")

    for _, row in df.sample(
        min(30, len(df)),
        random_state=42,
    ).iterrows():
        print(
            f"\n[{row['tweet_id']}] "
            f"{row['author_id']}"
        )
        print(row["text"])


if __name__ == "__main__":
    main()