from pathlib import Path
import sqlite3
import random


DB_PATH = Path("data/processed/twcs.db")

BRANDS = [
    "AmazonHelp",
    "Tesco",
]

NUM_THREADS = 8


def get_brand_tweet_ids(
    connection: sqlite3.Connection,
    brand: str,
) -> set[int]:
    rows = connection.execute(
        """
        SELECT tweet_id
        FROM tweets
        WHERE author_id = ?
        """,
        (brand,),
    ).fetchall()

    return {row[0] for row in rows}


def get_root(
    connection: sqlite3.Connection,
    tweet_id: int,
) -> int:
    """
    Walk backwards through parent links until we reach
    the root of the conversation.
    """

    current = tweet_id
    visited = set()

    while current not in visited:
        visited.add(current)

        row = connection.execute(
            """
            SELECT in_response_to_tweet_id
            FROM tweets
            WHERE tweet_id = ?
            """,
            (current,),
        ).fetchone()

        if row is None:
            break

        parent_id = row[0]

        if parent_id is None:
            break

        parent_exists = connection.execute(
            """
            SELECT 1
            FROM tweets
            WHERE tweet_id = ?
            """,
            (parent_id,),
        ).fetchone()

        if parent_exists is None:
            break

        current = parent_id

    return current


def get_thread(
    connection: sqlite3.Connection,
    root_id: int,
) -> list[tuple]:
    """
    Get all tweets belonging to the connected conversation
    rooted at root_id.
    """

    rows = connection.execute(
        """
        WITH RECURSIVE thread(tweet_id) AS (
            SELECT tweet_id
            FROM tweets
            WHERE tweet_id = ?

            UNION ALL

            SELECT t.tweet_id
            FROM tweets t
            JOIN thread r
              ON t.in_response_to_tweet_id = r.tweet_id
        )
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text
        FROM tweets
        WHERE tweet_id IN (
            SELECT tweet_id FROM thread
        )
        ORDER BY tweet_id
        """,
        (root_id,),
    ).fetchall()

    return rows


def main() -> None:
    connection = sqlite3.connect(DB_PATH)

    try:
        for brand in BRANDS:
            print("\n")
            print("=" * 100)
            print(f"BRAND: {brand}")
            print("=" * 100)

            brand_tweets = list(
                get_brand_tweet_ids(connection, brand)
            )

            random.seed(42)

            # Shuffle deterministically so our sample is reproducible.
            random.shuffle(brand_tweets)

            selected = []
            seen_roots = set()

            for tweet_id in brand_tweets:
                root = get_root(connection, tweet_id)

                if root in seen_roots:
                    continue

                thread = get_thread(connection, root)

                # We want meaningful multi-turn conversations.
                if len(thread) >= 4:
                    selected.append(thread)
                    seen_roots.add(root)

                if len(selected) >= NUM_THREADS:
                    break

            for index, thread in enumerate(selected, start=1):
                print("\n" + "-" * 100)
                print(f"THREAD {index} | {len(thread)} tweets")
                print("-" * 100)

                for (
                    tweet_id,
                    author_id,
                    inbound,
                    created_at,
                    text,
                ) in thread:
                    speaker = (
                        "CUSTOMER"
                        if inbound
                        else "BRAND"
                    )

                    print(
                        f"\n[{speaker}] "
                        f"{author_id} | "
                        f"{created_at}"
                    )
                    print(f"Tweet ID: {tweet_id}")
                    print(text)

    finally:
        connection.close()


if __name__ == "__main__":
    main()