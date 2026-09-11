from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_PATH = Path(
    "data/processed/tesco_cases.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/tesco_intent_clusters.parquet"
)

N_CLUSTERS = 20
TOP_TERMS = 12
EXAMPLES_PER_CLUSTER = 10


def main() -> None:
    print("=" * 90)
    print("TESCO INTENT DISCOVERY")
    print("=" * 90)

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nTotal cases: {len(df):,}")

    # ---------------------------------------------------------------
    # Intent discovery should use the INITIAL customer problem,
    # not follow-up messages or Tesco's responses.
    # ---------------------------------------------------------------

    df["intent_text"] = (
        df["initial_customer_message"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"https?://\S+",
            " ",
            regex=True,
        )
        .str.replace(
            r"@\w+",
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

    # Remove completely empty cases.
    df = df[
        df["intent_text"].str.len() >= 10
    ].copy()

    print(
        f"Cases with usable initial messages: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # TF-IDF representation.
    # ---------------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.90,
        max_features=30_000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(
        df["intent_text"]
    )

    print(
        f"\nTF-IDF matrix: "
        f"{X.shape[0]:,} documents × "
        f"{X.shape[1]:,} features"
    )

    # ---------------------------------------------------------------
    # K-Means.
    #
    # 20 clusters is deliberately an exploratory number.
    # It is NOT our final number of intents.
    # ---------------------------------------------------------------

    print(
        f"\nClustering into "
        f"{N_CLUSTERS} exploratory groups..."
    )

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X)

    df["cluster"] = labels

    terms = vectorizer.get_feature_names_out()

    # ---------------------------------------------------------------
    # Inspect clusters.
    # ---------------------------------------------------------------

    print("\n" + "=" * 90)
    print("CLUSTER RESULTS")
    print("=" * 90)

    for cluster_id in range(N_CLUSTERS):

        cluster_mask = (
            labels == cluster_id
        )

        cluster_df = df.loc[
            cluster_mask
        ]

        centroid = (
            model.cluster_centers_[cluster_id]
        )

        top_indices = centroid.argsort()[
            -TOP_TERMS:
        ][::-1]

        top_terms = [
            terms[index]
            for index in top_indices
        ]

        print("\n" + "-" * 90)
        print(
            f"CLUSTER {cluster_id} "
            f"| {len(cluster_df):,} cases"
        )
        print("-" * 90)

        print(
            "Top terms: "
            + ", ".join(top_terms)
        )

        # -----------------------------------------------------------
        # Representative examples.
        #
        # TF-IDF rows are L2-normalized by default, so dot product
        # with the centroid gives a useful similarity ranking.
        # We keep this sparse and memory-safe.
        # -----------------------------------------------------------

        positions = (
            cluster_df.index
            .to_series()
            .map(
                lambda index:
                df.index.get_loc(index)
            )
            .to_numpy()
        )

        cluster_vectors = X[positions]

        scores = (
            cluster_vectors
            .dot(centroid)
        )

        scores = scores.toarray().ravel() if hasattr(
            scores, "toarray"
        ) else scores.ravel()

        best_positions = scores.argsort()[
            -EXAMPLES_PER_CLUSTER:
        ][::-1]

        print("\nRepresentative examples:")

        for position in best_positions:

            row = cluster_df.iloc[position]

            print(
                f"\n[{row['case_id']}]"
            )

            print(
                row["initial_customer_message"]
            )

    # ---------------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------------

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()