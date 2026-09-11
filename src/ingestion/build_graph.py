from pathlib import Path
import sqlite3
import sys

import pandas as pd
from tqdm import tqdm


DATASET_PATH = Path(
    r"C:\Users\umang\.cache\kagglehub\datasets"
    r"\thoughtvector\customer-support-on-twitter"
    r"\versions\10\twcs\twcs.csv"
)

OUTPUT_DIR = Path("data/processed")
DB_PATH = OUTPUT_DIR / "twcs.db"

CHUNK_SIZE = 50_000


def create_database(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS tweets")

    connection.execute(
        """
        CREATE TABLE tweets (
            tweet_id INTEGER PRIMARY KEY,
            author_id TEXT NOT NULL,
            inbound INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            response_tweet_id TEXT,
            in_response_to_tweet_id INTEGER
        )
        """
    )

    connection.commit()


def insert_chunk(
    connection: sqlite3.Connection,
    df: pd.DataFrame,
) -> None:
    rows = []

    for row in df.itertuples(index=False):
        parent_id = (
            None
            if pd.isna(row.in_response_to_tweet_id)
            else int(row.in_response_to_tweet_id)
        )

        response_ids = (
            None
            if pd.isna(row.response_tweet_id)
            else str(row.response_tweet_id)
        )

        rows.append(
            (
                int(row.tweet_id),
                str(row.author_id),
                int(bool(row.inbound)),
                str(row.created_at),
                str(row.text),
                response_ids,
                parent_id,
            )
        )

    connection.executemany(
        """
        INSERT INTO tweets (
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
            in_response_to_tweet_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()


def create_indexes(connection: sqlite3.Connection) -> None:
    print("\nCreating indexes...")

    connection.execute(
        """
        CREATE INDEX idx_author
        ON tweets(author_id)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_parent
        ON tweets(in_response_to_tweet_id)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_inbound
        ON tweets(inbound)
        """
    )

    connection.commit()


def main() -> None:
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILDING TWCS CONVERSATION GRAPH")
    print("=" * 80)

    print(f"\nInput:  {DATASET_PATH}")
    print(f"Output: {DB_PATH}")
    print(f"Chunk:  {CHUNK_SIZE:,} rows")

    if DB_PATH.exists():
        print("\nExisting database found. Rebuilding it.")

    connection = sqlite3.connect(DB_PATH)

    try:
        create_database(connection)

        reader = pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
        )

        total = 0

        for chunk_number, chunk in enumerate(
            tqdm(reader, desc="Importing tweets"),
            start=1,
        ):
            insert_chunk(connection, chunk)
            total += len(chunk)

        print(f"\nImported {total:,} tweets.")

        create_indexes(connection)

        count = connection.execute(
            "SELECT COUNT(*) FROM tweets"
        ).fetchone()[0]

        print(f"Database rows: {count:,}")

        print("\nDatabase created successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()