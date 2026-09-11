from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = ROOT / "data" / "retrieval" / "tesco_retrieval_corpus.parquet"


class TfidfRetriever:
    def __init__(self, corpus_path=CORPUS_PATH):
        self.corpus = pd.read_parquet(corpus_path)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.corpus["initial_customer_message"]
        )

    def retrieve(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        top_indices = scores.argsort()[::-1][:top_k]

        results = self.corpus.iloc[top_indices].copy()
        results["similarity"] = scores[top_indices]

        return results.reset_index(drop=True)


def main():
    print("Loading Tesco retrieval corpus...")
    retriever = TfidfRetriever()

    print(f"Corpus size: {len(retriever.corpus)}")
    print(f"Vocabulary size: {len(retriever.vectorizer.vocabulary_)}")

    query = input("\nEnter a customer message: ").strip()

    if not query:
        print("No query provided.")
        return

    results = retriever.retrieve(query, top_k=5)

    print("\n" + "=" * 80)
    print("TOP 5 HISTORICAL MATCHES")
    print("=" * 80)

    for i, row in results.iterrows():
        print(f"\n--- Match {i + 1} ---")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Case ID: {row['case_id']}")
        print(f"Customer: {row['initial_customer_message']}")
        print("\nTranscript:")
        print(row["transcript"])


if __name__ == "__main__":
    main()