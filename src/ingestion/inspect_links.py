from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    r"C:\Users\umang\.cache\kagglehub\datasets"
    r"\thoughtvector\customer-support-on-twitter"
    r"\versions\10\twcs\twcs.csv"
)


def main() -> None:
    df = pd.read_csv(
        DATASET_PATH,
        nrows=10_000,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    print("=" * 80)
    print("RESPONSE LINK INSPECTION")
    print("=" * 80)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nExamples of response_tweet_id:")
    print(
        df.loc[
            df["response_tweet_id"].notna(),
            ["tweet_id", "response_tweet_id"],
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\nExamples of in_response_to_tweet_id:")
    print(
        df.loc[
            df["in_response_to_tweet_id"].notna(),
            ["tweet_id", "in_response_to_tweet_id"],
        ]
        .head(30)
        .to_string(index=False)
    )

    # Check whether response_tweet_id ever contains multiple IDs.
    response_values = (
        df["response_tweet_id"]
        .dropna()
        .astype(str)
    )

    multi_response = response_values[
        response_values.str.contains(",")
    ]

    print("\nMultiple response IDs:")
    print(f"Count: {len(multi_response):,}")

    if len(multi_response):
        print(multi_response.head(20).to_string(index=False))

    # Show a few complete linked conversations.
    print("\nSample rows with links:")
    linked = df[
        df["response_tweet_id"].notna()
        | df["in_response_to_tweet_id"].notna()
    ]

    print(
        linked[
            [
                "tweet_id",
                "author_id",
                "inbound",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()