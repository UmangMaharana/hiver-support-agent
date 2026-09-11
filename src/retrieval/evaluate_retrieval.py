from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = ROOT / "data" / "retrieval" / "tesco_retrieval_corpus.parquet"
GOLDEN_PATH = ROOT / "data" / "golden" / "tesco_golden_250_labeled.csv"
CASES_PATH = ROOT / "data" / "processed" / "tesco_cases.parquet"

OUTPUT_DIR = ROOT / "data" / "retrieval"
OUTPUT_PATH = OUTPUT_DIR / "tfidf_retrieval_eval.csv"


def main():
    print("Loading retrieval corpus...")
    corpus = pd.read_parquet(CORPUS_PATH)

    print("Loading golden set...")
    golden = pd.read_csv(GOLDEN_PATH)

    print("Loading reconstructed cases...")
    cases = pd.read_parquet(CASES_PATH)

    # Build a case_id -> initial message mapping for the historical corpus.
    corpus = corpus.copy()
    golden = golden.copy()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    print("Fitting TF-IDF...")
    matrix = vectorizer.fit_transform(
        corpus["initial_customer_message"]
    )

    print(f"Corpus size: {len(corpus)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Golden queries: {len(golden)}")

    # Map each corpus case to its original case information.
    case_info = (
        cases[
            [
                "case_id",
                "initial_customer_message",
            ]
        ]
        .drop_duplicates("case_id")
        .set_index("case_id")
    )

    rows = []

    for _, query_row in golden.iterrows():
        query = str(query_row["initial_customer_message"])
        gold_intent = query_row["intent"]
        golden_id = query_row["golden_id"]

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).ravel()

        top_indices = scores.argsort()[::-1][:5]

        for rank, idx in enumerate(top_indices, start=1):
            retrieved = corpus.iloc[idx]

            rows.append(
                {
                    "golden_id": golden_id,
                    "gold_intent": gold_intent,
                    "query": query,
                    "rank": rank,
                    "retrieved_case_id": retrieved["case_id"],
                    "similarity": scores[idx],
                    "retrieved_message": retrieved[
                        "initial_customer_message"
                    ],
                }
            )

    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    # ------------------------------------------------------------
    # Retrieval quality metrics
    # ------------------------------------------------------------

    grouped = results.groupby("golden_id")

    top1_scores = grouped.first()["similarity"]

    top1_nonzero = (top1_scores > 0).mean()

    # Since the historical corpus does not have human intent labels,
    # we use the golden intent + a lightweight keyword/rule classifier
    # later for intent agreement. For this first baseline, report
    # retrieval coverage and similarity statistics only.
    top5_nonzero = (
        grouped["similarity"]
        .apply(lambda x: (x > 0).any())
        .mean()
    )

    top5_mean_similarity = (
        grouped["similarity"]
        .mean()
        .mean()
    )

    print("\n" + "=" * 80)
    print("TF-IDF RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"Golden queries:              {len(golden)}")
    print(f"Top-1 nonzero retrieval:     {top1_nonzero:.3f}")
    print(f"Top-5 nonzero retrieval:     {top5_nonzero:.3f}")
    print(f"Mean top-5 similarity:       {top5_mean_similarity:.3f}")
    print(f"Mean top-1 similarity:       {top1_scores.mean():.3f}")
    print(f"Median top-1 similarity:     {top1_scores.median():.3f}")

    print("\nTop 10 weakest top-1 matches:")

    weakest = (
        results[results["rank"] == 1]
        .sort_values("similarity")
        .head(10)
    )

    for _, row in weakest.iterrows():
        print("\n---")
        print(f"Golden ID: {row['golden_id']}")
        print(f"Gold intent: {row['gold_intent']}")
        print(f"Query: {row['query']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Retrieved: {row['retrieved_message']}")

    print(f"\nSaved evaluation results to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()