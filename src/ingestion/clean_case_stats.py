from collections import defaultdict
from pathlib import Path
import sqlite3
import statistics


DB_PATH = Path("data/processed/twcs.db")

BRAND = "Tesco"

MIN_TURNS = 2
MAX_TURNS = 30


def load_graph(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        """
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
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
        parent_id,
    ) in rows:
        tweets[tweet_id] = {
            "author_id": author_id,
            "inbound": bool(inbound),
            "created_at": created_at,
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


def main() -> None:
    print("=" * 90)
    print("CLEAN TESCO SUPPORT CASE ANALYSIS")
    print("=" * 90)

    connection = sqlite3.connect(DB_PATH)

    try:
        print("\nLoading graph...")
        tweets = load_graph(connection)
    finally:
        connection.close()

    print(f"Loaded {len(tweets):,} tweets.")

    children = build_children(tweets)

    # Find all Tesco-authored tweets.
    tesco_tweets = [
        tweet_id
        for tweet_id, tweet in tweets.items()
        if tweet["author_id"] == BRAND
    ]

    print(
        f"Tesco tweets: {len(tesco_tweets):,}"
    )

    # Find unique graph components containing Tesco.
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
        f"Unique connected components: "
        f"{len(components):,}"
    )

    # Analyze each component.
    clean_cases = []

    stats = {
        "has_customer": 0,
        "has_tesco": 0,
        "one_customer": 0,
        "multi_customer": 0,
        "too_short": 0,
        "too_long": 0,
        "clean": 0,
    }

    all_lengths = []

    for root, component in components.items():
        component_tweets = [
            tweets[tweet_id]
            for tweet_id in component
        ]

        authors = {
            tweet["author_id"]
            for tweet in component_tweets
        }

        customer_authors = {
            tweet["author_id"]
            for tweet in component_tweets
            if tweet["inbound"]
            and tweet["author_id"] != BRAND
        }

        has_tesco = BRAND in authors
        has_customer = len(customer_authors) > 0

        if has_customer:
            stats["has_customer"] += 1

        if has_tesco:
            stats["has_tesco"] += 1

        if len(customer_authors) == 1:
            stats["one_customer"] += 1
        elif len(customer_authors) > 1:
            stats["multi_customer"] += 1

        length = len(component)
        all_lengths.append(length)

        if length < MIN_TURNS:
            stats["too_short"] += 1
            continue

        if length > MAX_TURNS:
            stats["too_long"] += 1
            continue

        if (
            has_tesco
            and len(customer_authors) == 1
        ):
            stats["clean"] += 1

            customer_id = next(
                iter(customer_authors)
            )

            clean_cases.append(
                {
                    "root_id": root,
                    "customer_id": customer_id,
                    "turns": length,
                }
            )

    print("\n" + "-" * 90)
    print("COMPONENT ANALYSIS")
    print("-" * 90)

    print(
        f"Components with customer: "
        f"{stats['has_customer']:,}"
    )

    print(
        f"Components with exactly 1 customer: "
        f"{stats['one_customer']:,}"
    )

    print(
        f"Components with multiple customers: "
        f"{stats['multi_customer']:,}"
    )

    print(
        f"Components too short (<{MIN_TURNS}): "
        f"{stats['too_short']:,}"
    )

    print(
        f"Components too long (>{MAX_TURNS}): "
        f"{stats['too_long']:,}"
    )

    print(
        f"\nCLEAN CASES: "
        f"{stats['clean']:,}"
    )

    if clean_cases:
        lengths = [
            case["turns"]
            for case in clean_cases
        ]

        print(
            f"Clean case avg turns: "
            f"{statistics.mean(lengths):.2f}"
        )

        print(
            f"Clean case median turns: "
            f"{statistics.median(lengths):.0f}"
        )

        print(
            f"Clean cases >=3 turns: "
            f"{sum(x >= 3 for x in lengths) / len(lengths) * 100:.1f}%"
        )

        print(
            f"Clean cases >=4 turns: "
            f"{sum(x >= 4 for x in lengths) / len(lengths) * 100:.1f}%"
        )

        print(
            f"Clean case max turns: "
            f"{max(lengths):,}"
        )

    print("\n" + "-" * 90)
    print("SAMPLE CLEAN CASE ROOTS")
    print("-" * 90)

    for case in sorted(
        clean_cases,
        key=lambda x: x["turns"],
        reverse=True,
    )[:20]:
        print(
            f"Root={case['root_id']:<10} "
            f"Customer={case['customer_id']:<10} "
            f"Turns={case['turns']}"
        )


if __name__ == "__main__":
    main()