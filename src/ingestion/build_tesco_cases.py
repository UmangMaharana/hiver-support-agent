from collections import defaultdict
from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("data/processed/twcs.db")
OUTPUT_PATH = Path("data/processed/tesco_cases.parquet")

BRAND = "Tesco"

MIN_TURNS = 2
MAX_TURNS = 30


def load_tweets(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        """
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            in_response_to_tweet_id
        FROM tweets
        """
    ).fetchall()

    tweets = {}

    for (
        tweet_id,
        author_id,
        inbound,
        created_at,
        text,
        parent_id,
    ) in rows:
        tweets[tweet_id] = {
            "author_id": author_id,
            "inbound": bool(inbound),
            "created_at": str(created_at),
            "text": str(text),
            "parent_id": parent_id,
        }

    return tweets


def build_children(tweets: dict) -> dict[int, list[int]]:
    children = defaultdict(list)

    for tweet_id, tweet in tweets.items():
        parent_id = tweet["parent_id"]

        if (
            parent_id is not None
            and parent_id in tweets
        ):
            children[parent_id].append(tweet_id)

    return children


def find_root(
    tweet_id: int,
    tweets: dict,
) -> int:
    current = tweet_id
    visited = set()

    while current not in visited:
        visited.add(current)

        parent_id = tweets[current]["parent_id"]

        if (
            parent_id is None
            or parent_id not in tweets
        ):
            break

        current = parent_id

    return current


def collect_component(
    root_id: int,
    tweets: dict,
    children: dict,
) -> set[int]:
    component = set()
    stack = [root_id]

    while stack:
        current = stack.pop()

        if current in component:
            continue

        component.add(current)

        for child in children.get(current, []):
            stack.append(child)

    return component


def make_transcript(
    tweet_ids: list[int],
    tweets: dict,
) -> str:
    lines = []

    for tweet_id in tweet_ids:
        tweet = tweets[tweet_id]

        speaker = (
            "CUSTOMER"
            if tweet["inbound"]
            else "TESCO"
        )

        text = tweet["text"].strip()

        if not text:
            continue

        lines.append(
            f"{speaker}: {text}"
        )

    return "\n\n".join(lines)


def main() -> None:
    print("=" * 90)
    print("BUILDING CLEAN TESCO SUPPORT CASES")
    print("=" * 90)

    connection = sqlite3.connect(DB_PATH)

    try:
        print("\nLoading graph...")
        tweets = load_tweets(connection)
    finally:
        connection.close()

    print(f"Loaded {len(tweets):,} tweets.")

    children = build_children(tweets)

    tesco_tweets = [
        tweet_id
        for tweet_id, tweet in tweets.items()
        if tweet["author_id"] == BRAND
    ]

    print(
        f"Tesco tweets: {len(tesco_tweets):,}"
    )

    # ---------------------------------------------------------------
    # Discover connected components.
    # ---------------------------------------------------------------

    components = {}

    for tweet_id in tesco_tweets:
        root = find_root(
            tweet_id,
            tweets,
        )

        if root not in components:
            components[root] = collect_component(
                root,
                tweets,
                children,
            )

    print(
        f"Connected components: "
        f"{len(components):,}"
    )

    # ---------------------------------------------------------------
    # Build clean cases.
    # ---------------------------------------------------------------

    cases = []

    for root_id, component in components.items():

        component_tweets = [
            tweets[tweet_id]
            for tweet_id in component
        ]

        customer_ids = {
            tweet["author_id"]
            for tweet in component_tweets
            if tweet["inbound"]
            and tweet["author_id"] != BRAND
        }

        # Precision-first filtering.
        if len(customer_ids) != 1:
            continue

        if not any(
            tweet["author_id"] == BRAND
            for tweet in component_tweets
        ):
            continue

        if not any(
            tweet["inbound"]
            for tweet in component_tweets
        ):
            continue

        if (
            len(component) < MIN_TURNS
            or len(component) > MAX_TURNS
        ):
            continue

        # -----------------------------------------------------------
        # IMPORTANT:
        # Sort chronologically by created_at.
        # -----------------------------------------------------------

        ordered_ids = sorted(
            component,
            key=lambda tweet_id: (
                tweets[tweet_id]["created_at"],
                tweet_id,
            ),
        )

        ordered_tweets = [
            tweets[tweet_id]
            for tweet_id in ordered_ids
        ]

        customer_turns = sum(
            tweet["inbound"]
            for tweet in ordered_tweets
        )

        tesco_turns = sum(
            not tweet["inbound"]
            for tweet in ordered_tweets
        )

        customer_id = next(
            iter(customer_ids)
        )

        transcript = make_transcript(
            ordered_ids,
            tweets,
        )

        initial_customer_message = next(
            (
                tweet["text"].strip()
                for tweet in ordered_tweets
                if tweet["inbound"]
                and tweet["text"].strip()
            ),
            "",
        )

        final_customer_message = next(
            (
                tweet["text"].strip()
                for tweet in reversed(ordered_tweets)
                if tweet["inbound"]
                and tweet["text"].strip()
            ),
            "",
        )

        cases.append(
            {
                "case_id": f"TESCO_{root_id}",
                "root_tweet_id": root_id,
                "customer_id": customer_id,
                "created_at": ordered_tweets[0]["created_at"],
                "turn_count": len(ordered_tweets),
                "customer_turns": int(customer_turns),
                "tesco_turns": int(tesco_turns),
                "initial_customer_message": initial_customer_message,
                "final_customer_message": final_customer_message,
                "transcript": transcript,
            }
        )

    # ---------------------------------------------------------------
    # Save.
    # ---------------------------------------------------------------

    df = pd.DataFrame(cases)

    df = df.sort_values(
        "created_at"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Report.
    # ---------------------------------------------------------------

    print("\n" + "=" * 90)
    print("TESCO SUPPORT CASE DATASET")
    print("=" * 90)

    print(
        f"\nClean cases:       {len(df):,}"
    )

    print(
        f"Unique customers:  "
        f"{df['customer_id'].nunique():,}"
    )

    print(
        f"Average turns:     "
        f"{df['turn_count'].mean():.2f}"
    )

    print(
        f"Median turns:      "
        f"{df['turn_count'].median():.0f}"
    )

    print(
        f"Output:            {OUTPUT_PATH}"
    )

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nSample cases:")

    for _, row in df.sample(
        min(5, len(df)),
        random_state=42,
    ).iterrows():

        print("\n" + "-" * 90)
        print(
            f"{row['case_id']} | "
            f"{row['turn_count']} turns"
        )
        print("-" * 90)
        print(row["transcript"])


if __name__ == "__main__":
    main()