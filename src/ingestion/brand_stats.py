from collections import Counter

import pandas as pd


DATASET_PATH = (
    r"C:\Users\umang\.cache\kagglehub\datasets"
    r"\thoughtvector\customer-support-on-twitter"
    r"\versions\10\twcs\twcs.csv"
)

CHUNK_SIZE = 100_000

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


def main() -> None:
    stats = {
        brand: {
            "total": 0,
            "inbound": 0,
            "outbound": 0,
            "customers": set(),
            "support_replies_with_response": 0,
            "support_replies_without_response": 0,
        }
        for brand in CANDIDATES
    }

    print("Analyzing candidate brands...")
    print()

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
            usecols=[
                "author_id",
                "inbound",
                "response_tweet_id",
            ],
        ),
        start=1,
    ):
        for brand in CANDIDATES:
            brand_mask = df["author_id"].eq(brand)

            brand_df = df.loc[brand_mask]

            if brand_df.empty:
                continue

            stats[brand]["total"] += len(brand_df)

            inbound = brand_df["inbound"]
            stats[brand]["inbound"] += int(inbound.sum())
            stats[brand]["outbound"] += int((~inbound).sum())

            # For inbound tweets, author_id is the customer.
            stats[brand]["customers"].update(
                brand_df.loc[inbound, "author_id"]
                .dropna()
                .astype(str)
            )

            # An outbound support tweet with a response_tweet_id
            # indicates that the support reply has a known follow-up.
            outbound = brand_df.loc[~inbound]

            has_response = outbound["response_tweet_id"].notna()

            stats[brand]["support_replies_with_response"] += int(
                has_response.sum()
            )
            stats[brand]["support_replies_without_response"] += int(
                (~has_response).sum()
            )

        if chunk_number % 5 == 0:
            print(f"Processed {chunk_number * CHUNK_SIZE:,} rows...")

    print("\n" + "=" * 80)
    print("CANDIDATE BRAND ANALYSIS")
    print("=" * 80)

    print(
        f"\n{'Brand':<18}"
        f"{'Total':>10}"
        f"{'Inbound':>10}"
        f"{'Outbound':>10}"
        f"{'Customers':>12}"
        f"{'Response %':>12}"
    )

    print("-" * 80)

    for brand, s in sorted(
        stats.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    ):
        total = s["total"]
        outbound = s["outbound"]
        with_response = s["support_replies_with_response"]

        response_rate = (
            (with_response / outbound * 100)
            if outbound
            else 0
        )

        print(
            f"{brand:<18}"
            f"{total:>10,}"
            f"{s['inbound']:>10,}"
            f"{outbound:>10,}"
            f"{len(s['customers']):>12,}"
            f"{response_rate:>11.1f}%"
        )


if __name__ == "__main__":
    main()