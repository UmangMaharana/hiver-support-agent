import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generation.build_prompt import build_prompt


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "tesco_golden_250_labeled.csv"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "baselines"
    / "baseline_predictions.csv"
)

RETRIEVAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval"
    / "tesco_retrieval_corpus.parquet"
)

RESOLUTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval"
    / "tesco_resolution_corpus.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generation"
    / "reply_predictions.csv"
)


MODEL_NAME = os.getenv(
    "GENERATION_MODEL",
    "gemini-2.5-flash",
)

TOP_K = 3

# Gemini/network resilience
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 3

# SDK-level timeout in milliseconds.
REQUEST_TIMEOUT_MS = 30_000

# Additional application-level timeout.
GENERATION_TIMEOUT_SECONDS = 35


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Add it to your .env file."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
        ),
    )


def load_retrieval_index():
    print("Retrieval corpus: loading...")

    retrieval_df = pd.read_parquet(
        RETRIEVAL_PATH
    )

    resolution_df = pd.read_parquet(
        RESOLUTION_PATH
    )

    resolution_cols = [
        "case_id",
        "resolution",
        "resolution_source",
    ]

    resolution_df = resolution_df[
        resolution_cols
    ].copy()

    retrieval_df = retrieval_df.merge(
        resolution_df,
        on="case_id",
        how="left",
    )

    retrieval_df[
        "initial_customer_message"
    ] = (
        retrieval_df[
            "initial_customer_message"
        ]
        .fillna("")
        .astype(str)
    )

    retrieval_df["resolution"] = (
        retrieval_df["resolution"]
        .fillna("")
        .astype(str)
    )

    retrieval_df["resolution_source"] = (
        retrieval_df["resolution_source"]
        .fillna("")
        .astype(str)
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        retrieval_df[
            "initial_customer_message"
        ]
    )

    print(
        f"Retrieval corpus: "
        f"{len(retrieval_df)}"
    )

    print(
        "Cases with resolution guidance: "
        f"{retrieval_df['resolution'].ne('').sum()}"
    )

    print(
        "Retrieval vocabulary: "
        f"{len(vectorizer.vocabulary_)}"
    )

    return (
        retrieval_df,
        vectorizer,
        matrix,
    )


def retrieve_cases(
    query: str,
    retrieval_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    top_indices = (
        similarities
        .argsort()[::-1][:top_k]
    )

    results = []

    for idx in top_indices:

        row = retrieval_df.iloc[idx]

        results.append(
            {
                "case_id": str(
                    row["case_id"]
                ),
                "similarity": float(
                    similarities[idx]
                ),
                "retrieved_message": str(
                    row[
                        "initial_customer_message"
                    ]
                ),
                "resolution": str(
                    row["resolution"]
                ),
                "resolution_source": str(
                    row["resolution_source"]
                ),
            }
        )

    return results


def load_inputs():

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    baseline = baseline[
        [
            "golden_id",
            "logistic_prediction",
        ]
    ].copy()

    baseline = baseline.rename(
        columns={
            "logistic_prediction":
                "predicted_intent"
        }
    )

    inputs = golden.merge(
        baseline,
        on="golden_id",
        how="inner",
    )

    return inputs


def generate_one(
    client: genai.Client,
    customer_message: str,
    predicted_intent: str,
    retrieved_cases: list[dict[str, Any]],
) -> dict[str, Any]:

    prompt = build_prompt(
        customer_message=customer_message,
        predicted_intent=predicted_intent,
        retrieved_cases=retrieved_cases,
    )

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "reply": {
                "type": "STRING",
            },
            "escalate": {
                "type": "BOOLEAN",
            },
            "escalation_reason": {
                "type": "STRING",
            },
        },
        "required": [
            "reply",
            "escalate",
            "escalation_reason",
        ],
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        print(
            f"  Gemini request "
            f"{attempt}/{MAX_RETRIES}..."
        )

        def call_gemini():

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type=(
                        "application/json"
                    ),
                    response_schema=(
                        response_schema
                    ),
                ),
            )

        executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            )
        )

        future = executor.submit(
            call_gemini
        )

        try:

            response = future.result(
                timeout=GENERATION_TIMEOUT_SECONDS
            )

            # Request completed normally.
            executor.shutdown(
                wait=True
            )

        except concurrent.futures.TimeoutError:

            # Do not wait for a stuck HTTP request.
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

            raise TimeoutError(
                "Gemini request exceeded "
                f"{GENERATION_TIMEOUT_SECONDS} "
                "seconds."
            )

        except KeyboardInterrupt:

            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

            raise

        except Exception:

            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

            raise

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty "
                "response."
            )

        try:

            result = json.loads(
                response.text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Gemini returned invalid JSON: "
                f"{exc}"
            ) from exc

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Gemini response was not "
                "a JSON object."
            )

        reply = str(
            result.get(
                "reply",
                "",
            )
        ).strip()

        escalate = bool(
            result.get(
                "escalate",
                False,
            )
        )

        escalation_reason = str(
            result.get(
                "escalation_reason",
                "",
            )
        ).strip()

        if not reply:

            raise RuntimeError(
                "Gemini returned an empty "
                "reply."
            )

        return {
            "reply": reply,
            "escalate": escalate,
            "escalation_reason": (
                escalation_reason
            ),
        }

    raise RuntimeError(
        f"Gemini failed after "
        f"{MAX_RETRIES} attempts: "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    )


def main():

    print("MAIN STARTED")

    print("=" * 80)
    print("TESCO REPLY GENERATION")
    print("=" * 80)

    client = get_client()

    inputs = load_inputs()

    (
        retrieval_df,
        vectorizer,
        matrix,
    ) = load_retrieval_index()

    print(
        f"Golden examples: "
        f"{len(inputs)}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Examples with predictions: "
        f"{len(inputs)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_PATH.exists():

        existing = pd.read_csv(
            OUTPUT_PATH
        )

        existing_ids = set(
            existing[
                "golden_id"
            ].astype(str)
        )

    else:

        existing = pd.DataFrame()

        existing_ids = set()

    print(
        "Existing generated examples: "
        f"{len(existing_ids)}"
    )

    generated_rows = []

    if not existing.empty:

        generated_rows = (
            existing.to_dict(
                "records"
            )
        )

    total = len(inputs)

    for position, (_, row) in enumerate(
        inputs.iterrows(),
        start=1,
    ):

        golden_id = str(
            row["golden_id"]
        )

        if golden_id in existing_ids:

            print(
                f"[{position}/{total}] "
                f"{golden_id} "
                "— already generated"
            )

            continue

        customer_message = str(
            row[
                "initial_customer_message"
            ]
        )

        predicted_intent = str(
            row[
                "predicted_intent"
            ]
        )

        print()

        print(
            f"[{position}/{total}] "
            f"{golden_id}"
        )

        print(
            f"Intent: "
            f"{predicted_intent}"
        )

        print(
            f"Customer: "
            f"{customer_message}"
        )

        retrieved_cases = retrieve_cases(
            query=customer_message,
            retrieval_df=retrieval_df,
            vectorizer=vectorizer,
            matrix=matrix,
            top_k=TOP_K,
        )

        print()
        print("Retrieved:")

        for rank, case in enumerate(
            retrieved_cases,
            start=1,
        ):

            print(
                f"\n  [{rank}] "
                f"similarity="
                f"{case['similarity']:.3f}"
            )

            print(
                "  Customer: "
                f"{case['retrieved_message'][:200]}"
            )

            print(
                "  Resolution: "
                f"{case['resolution'][:300]}"
            )

            print(
                "  Source: "
                f"{case['resolution_source']}"
            )

        try:

            result = generate_one(
                client=client,
                customer_message=(
                    customer_message
                ),
                predicted_intent=(
                    predicted_intent
                ),
                retrieved_cases=(
                    retrieved_cases
                ),
            )

            output_row = {
                "golden_id": golden_id,
                "case_id": str(
                    row["case_id"]
                ),
                "gold_intent": str(
                    row["intent"]
                ),
                "predicted_intent": (
                    predicted_intent
                ),
                "customer_message": (
                    customer_message
                ),
                "reply": result[
                    "reply"
                ],
                "escalate": result[
                    "escalate"
                ],
                "escalation_reason": (
                    result[
                        "escalation_reason"
                    ]
                ),
            }

            generated_rows.append(
                output_row
            )

            # Save after every
            # successful generation.
            pd.DataFrame(
                generated_rows
            ).to_csv(
                OUTPUT_PATH,
                index=False,
            )

            existing_ids.add(
                golden_id
            )

            print()

            print(
                f"Reply: "
                f"{result['reply']}"
            )

            print(
                f"Escalate: "
                f"{result['escalate']}"
            )

            print(
                "Reason: "
                f"{result['escalation_reason']}"
            )

        except KeyboardInterrupt:

            print()
            print(
                "Generation interrupted "
                "by user."
            )

            raise

        except Exception as exc:

            print()
            print(
                f"!!! FAILED "
                f"{golden_id}"
            )

            print(
                f"!!! "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                "!!! Skipping this "
                "example and continuing."
            )

            failure_row = {
                "golden_id": golden_id,
                "case_id": str(
                    row["case_id"]
                ),
                "gold_intent": str(
                    row["intent"]
                ),
                "predicted_intent": (
                    predicted_intent
                ),
                "customer_message": (
                    customer_message
                ),
                "reply": "",
                "escalate": "",
                "escalation_reason": (
                    "GENERATION_FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

            generated_rows.append(
                failure_row
            )

            pd.DataFrame(
                generated_rows
            ).to_csv(
                OUTPUT_PATH,
                index=False,
            )

            # Failed examples are
            # intentionally NOT added
            # to existing_ids.
            #
            # A future run will retry
            # them.

            continue

    print()

    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

    final_df = pd.DataFrame(
        generated_rows
    )

    successful = (
        final_df["reply"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    failed = (
        len(final_df)
        - successful
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


if __name__ == "__main__":
    main()