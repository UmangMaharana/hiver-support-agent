from collections import defaultdict
from pathlib import Path

import sqlite3
import statistics


DB_PATH = Path("data/processed/twcs.db")

CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
]


def load_tweets(connection: sqlite3.Connection) -> dict:
    """
    Load the lightweight graph information needed for analysis.

    Returns:
        tweet_id -> {
            author_id,
            inbound,
            parent_id,
        }
    """
    rows = connection.execute(
        """
        SELECT
            tweet_id,
            author_id,
            inbound,
            in_response_to_tweet_id
        FROM tweets
        """
    ).fetchall()

    tweets = {}

    for tweet_id, author_id, inbound, parent_id in rows:
        tweets[tweet_id] = {
            "author_id": author_id,
            "inbound": bool(inbound),
            "parent_id": parent_id,
        }

    return tweets


def build_children(tweets: dict) -> dict:
    """Build parent -> children relationships."""
    children = defaultdict(list)

    for tweet_id, tweet in tweets.items():
        parent_id = tweet["parent_id"]

        if parent_id is not None and parent_id in tweets:
            children[parent_id].append(tweet_id)

    return children


def find_brand_threads(
    brand: str,
    tweets: dict,
    children: dict,
) -> list[list[int]]:
    """
    Find conversation components containing the target brand.

    We start from brand-authored tweets and walk upward to the root.
    Then we walk downward through connected replies.
    """

    brand_tweets = {
        tweet_id
        for tweet_id, tweet in tweets.items()
        if tweet["author_id"] == brand
    }

    roots = set()

    for tweet_id in brand_tweets:
        current = tweet_id

        while (
            tweets[current]["parent_id"] is not None
            and tweets[current]["parent_id"] in tweets
        ):
            current = tweets[current]["parent_id"]

        roots.add(current)

    threads = []

    for root in roots:
        thread = []
        stack = [root]
        visited = set()

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            thread.append(current)

            for child in children.get(current, []):
                stack.append(child)

        # Only keep threads that actually contain the brand.
        if any(tweet_id in brand_tweets for tweet_id in thread):
            threads.append(thread)

    return threads


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    print("=" * 80)
    print("CONVERSATION QUALITY ANALYSIS")
    print("=" * 80)

    print("\nLoading graph...")

    connection = sqlite3.connect(DB_PATH)

    try:
        tweets = load_tweets(connection)
    finally:
        connection.close()

    print(f"Loaded {len(tweets):,} tweets.")

    children = build_children(tweets)

    print("Built reply graph.")

    print(
        f"\n{'Brand':<18}"
        f"{'Brand Tweets':>14}"
        f"{'Threads':>12}"
        f"{'Avg Turns':>12}"
        f"{'Median':>10}"
        f"{'>=3':>10}"
        f"{'>=4':>10}"
        f"{'Max':>10}"
    )

    print("-" * 96)

    results = {}

    for brand in CANDIDATES:
        threads = find_brand_threads(
            brand,
            tweets,
            children,
        )

        # A thread may include unrelated branches, but for now
        # we're measuring the connected support conversation.
        lengths = [len(thread) for thread in threads]

        brand_tweet_count = sum(
            1
            for tweet in tweets.values()
            if tweet["author_id"] == brand
        )

        if lengths:
            avg_turns = statistics.mean(lengths)
            median_turns = statistics.median(lengths)
            pct_3 = sum(x >= 3 for x in lengths) / len(lengths) * 100
            pct_4 = sum(x >= 4 for x in lengths) / len(lengths) * 100
            max_turns = max(lengths)
        else:
            avg_turns = 0
            median_turns = 0
            pct_3 = 0
            pct_4 = 0
            max_turns = 0

        results[brand] = {
            "brand_tweets": brand_tweet_count,
            "threads": len(threads),
            "avg": avg_turns,
            "median": median_turns,
            "pct_3": pct_3,
            "pct_4": pct_4,
            "max": max_turns,
        }

        print(
            f"{brand:<18}"
            f"{brand_tweet_count:>14,}"
            f"{len(threads):>12,}"
            f"{avg_turns:>11.2f}"
            f"{median_turns:>10.0f}"
            f"{pct_3:>9.1f}%"
            f"{pct_4:>9.1f}%"
            f"{max_turns:>10,}"
        )

    print("\n")
    print("Interpretation:")
    print("- Higher thread count = more potential training/evaluation cases.")
    print("- Higher >=3 / >=4 percentages = richer multi-turn support history.")
    print("- Longer is not automatically better; extremely long threads may be noisy.")


if __name__ == "__main__":
    main()