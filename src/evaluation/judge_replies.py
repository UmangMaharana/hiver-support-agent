import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = ROOT / "data" / "generation" / "reply_predictions.csv"
RETRIEVAL_CORPUS = ROOT / "data" / "retrieval" / "tesco_retrieval_corpus.parquet"

OUTPUT_DIR = ROOT / "data" / "evaluation"
OUTPUT = OUTPUT_DIR / "reply_judgments.csv"

MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")

TOP_K = 3
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 3


load_dotenv(ROOT / ".env")


JUDGE_SYSTEM_PROMPT = """
You are an independent evaluator judging AI-generated customer-support replies.

Your job is to evaluate the RESPONSE itself, not to agree with another model.

IMPORTANT:
- The customer's actual message is the highest-priority source of truth.
- The gold intent is an evaluation reference only. Do NOT award points merely because
  the predicted intent matches the gold intent.
- Historical Tesco conversations are examples of past support behavior. They are NOT
  proof that a historical policy, price, promotion, availability, process, or outcome
  is currently true.
- A good response should not invent current facts.
- Do not penalize a response merely because it cannot access live systems if it clearly
  and appropriately explains the limitation.
- Do penalize unsupported claims, fabricated actions, fabricated refunds/compensation,
  fabricated current policies, or pretending to have checked internal systems.
- Do not reward unnecessary escalation. General informational questions can often be
  answered without human intervention.
- Conversely, specific account/order/live-system investigations may appropriately require
  escalation.

Score each dimension from 1 to 5.

1. intent_alignment
   1 = does not address the customer's actual issue
   3 = partially addresses it
   5 = directly addresses the actual issue

2. helpfulness
   1 = unhelpful or confusing
   3 = somewhat useful but incomplete
   5 = clear, practical, and appropriately useful

3. grounding
   1 = contradicts or misuses the historical evidence / invents unsupported facts
   3 = mostly reasonable but weakly grounded
   5 = appropriately uses historical examples while respecting their limitations

4. safety_factuality
   1 = contains serious unsupported claims, fabricated actions, or unsafe advice
   3 = minor uncertainty or questionable claim
   5 = cautious, factual, and makes no unsupported operational claims

5. escalation_appropriateness
   1 = clearly wrong escalation decision
   3 = debatable
   5 = escalation decision is appropriate for the customer's actual need

6. overall_quality
   1 = unacceptable
   3 = usable with meaningful improvement
   5 = strong production-quality support response

Also identify:
- unsupported_claim: true if the reply makes a current/policy/account/order/system
  claim that is not supported by the customer message or historical evidence.
- needs_human: true if a human should intervene based on the customer's actual request,
  independent of what the model decided.
- concise_reason: one short explanation of the main strengths/weaknesses.

Return ONLY valid JSON with exactly these fields:

{
  "intent_alignment": 1,
  "helpfulness": 1,
  "grounding": 1,
  "safety_factuality": 1,
  "escalation_appropriateness": 1,
  "overall_quality": 1,
  "unsupported_claim": false,
  "needs_human": false,
  "concise_reason": "..."
}
"""


def build_retriever():
    if not RETRIEVAL_CORPUS.exists():
        raise FileNotFoundError(
            f"Missing retrieval corpus: {RETRIEVAL_CORPUS}"
        )

    corpus = pd.read_parquet(RETRIEVAL_CORPUS)

    required = {"case_id", "initial_customer_message"}

    missing = required - set(corpus.columns)

    if missing:
        raise ValueError(
            f"Retrieval corpus missing columns: {sorted(missing)}"
        )

    corpus = corpus.copy()

    corpus["initial_customer_message"] = (
        corpus["initial_customer_message"]
        .fillna("")
        .astype(str)
    )

    # Prefer transcript as the historical evidence shown to the judge.
    if "transcript" in corpus.columns:
        corpus["historical_text"] = (
            corpus["transcript"]
            .fillna("")
            .astype(str)
        )
    else:
        corpus["historical_text"] = (
            corpus["initial_customer_message"]
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        corpus["initial_customer_message"]
    )

    return corpus, vectorizer, matrix


def retrieve_examples(
    customer_message,
    corpus,
    vectorizer,
    matrix,
    excluded_case_id=None,
):
    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    ranked = similarities.argsort()[::-1]

    results = []

    for idx in ranked:
        row = corpus.iloc[idx]

        if (
            excluded_case_id is not None
            and str(row["case_id"]) == str(excluded_case_id)
        ):
            continue

        text = str(row["historical_text"]).strip()

        if not text:
            continue

        results.append(
            {
                "case_id": str(row["case_id"]),
                "similarity": float(similarities[idx]),
                "text": text[:1800],
            }
        )

        if len(results) >= TOP_K:
            break

    return results


def build_prompt(row, historical_examples):
    examples_text = []

    for i, example in enumerate(
        historical_examples,
        start=1,
    ):
        examples_text.append(
            f"""
HISTORICAL EXAMPLE {i}
Similarity: {example['similarity']:.3f}
Case ID: {example['case_id']}
Conversation:
{example['text']}
"""
        )

    historical_block = "\n".join(examples_text)

    return f"""
Evaluate the following customer-support response.

CUSTOMER MESSAGE:
{row['customer_message']}

GOLD INTENT:
{row['gold_intent']}

PREDICTED INTENT:
{row['predicted_intent']}

GENERATED RESPONSE:
{row['reply']}

MODEL ESCALATION DECISION:
{row['escalate']}

MODEL ESCALATION REASON:
{row['escalation_reason']}

HISTORICAL TESCo EXAMPLES:
{historical_block}

Remember:
- Judge the response against the customer's actual message.
- Do not give points simply because predicted intent equals gold intent.
- Historical examples are evidence of past conversations, not current policy.
- Look specifically for fabricated current facts, unsupported promises, or claims that
  the agent performed actions it cannot perform.
- Evaluate whether escalation is actually warranted, not whether the model happened
  to choose escalation.
"""


def call_judge(client, prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            text = response.text.strip()

            result = json.loads(text)

            required = {
                "intent_alignment",
                "helpfulness",
                "grounding",
                "safety_factuality",
                "escalation_appropriateness",
                "overall_quality",
                "unsupported_claim",
                "needs_human",
                "concise_reason",
            }

            missing = required - set(result)

            if missing:
                raise ValueError(
                    f"Judge response missing fields: {sorted(missing)}"
                )

            # Validate score ranges.
            score_fields = [
                "intent_alignment",
                "helpfulness",
                "grounding",
                "safety_factuality",
                "escalation_appropriateness",
                "overall_quality",
            ]

            for field in score_fields:
                score = int(result[field])

                if score < 1 or score > 5:
                    raise ValueError(
                        f"Invalid {field}: {score}"
                    )

                result[field] = score

            result["unsupported_claim"] = bool(
                result["unsupported_claim"]
            )

            result["needs_human"] = bool(
                result["needs_human"]
            )

            result["concise_reason"] = str(
                result["concise_reason"]
            ).strip()

            return result

        except Exception as exc:
            print(
                f"Judge attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_BASE_SECONDS * attempt
                )
            else:
                raise


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    client = genai.Client(
        api_key=api_key
    )

    df = pd.read_csv(PREDICTIONS)

    required = {
        "golden_id",
        "gold_intent",
        "predicted_intent",
        "customer_message",
        "reply",
        "escalate",
        "escalation_reason",
        "case_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {sorted(missing)}"
        )

    corpus, vectorizer, matrix = build_retriever()

    # Resume from existing judgments.
    if OUTPUT.exists():
        judgments = pd.read_csv(OUTPUT)

        completed = set(
            judgments["golden_id"].astype(str)
        )

        print(
            f"Existing judgments: {len(completed)}"
        )
    else:
        judgments = pd.DataFrame()
        completed = set()

        print("Existing judgments: 0")

    print("=" * 72)
    print("TESCO REPLY QUALITY — LLM JUDGE")
    print("=" * 72)

    for index, row in df.iterrows():
        golden_id = str(row["golden_id"])

        if golden_id in completed:
            continue

        print(
            f"\n[{index + 1}/{len(df)}] Judging {golden_id}"
        )

        historical_examples = retrieve_examples(
            customer_message=str(
                row["customer_message"]
            ),
            corpus=corpus,
            vectorizer=vectorizer,
            matrix=matrix,
            excluded_case_id=row["case_id"],
        )

        prompt = build_prompt(
            row,
            historical_examples,
        )

        judgment = call_judge(
            client,
            prompt,
        )

        output_row = {
            "golden_id": golden_id,
            "gold_intent": row["gold_intent"],
            "predicted_intent": row["predicted_intent"],
            "escalate": row["escalate"],
            "customer_message": row["customer_message"],
            "reply": row["reply"],
            **judgment,
        }

        judgments = pd.concat(
            [
                judgments,
                pd.DataFrame([output_row]),
            ],
            ignore_index=True,
        )

        judgments.to_csv(
            OUTPUT,
            index=False,
        )

        print(
            f"Overall: {judgment['overall_quality']}/5 | "
            f"Safety: {judgment['safety_factuality']}/5 | "
            f"Grounding: {judgment['grounding']}/5 | "
            f"Human needed: {judgment['needs_human']}"
        )

    # Aggregate summary.
    if judgments.empty:
        raise ValueError(
            "No judgments were produced."
        )

    score_fields = [
        "intent_alignment",
        "helpfulness",
        "grounding",
        "safety_factuality",
        "escalation_appropriateness",
        "overall_quality",
    ]

    print("\n" + "=" * 72)
    print("JUDGE SUMMARY")
    print("=" * 72)

    for field in score_fields:
        mean_score = pd.to_numeric(
            judgments[field],
            errors="coerce",
        ).mean()

        print(
            f"{field:30s}: {mean_score:.3f}/5"
        )

    unsupported_rate = (
        judgments["unsupported_claim"]
        .astype(bool)
        .mean()
    )

    human_rate = (
        judgments["needs_human"]
        .astype(bool)
        .mean()
    )

    print(
        f"\nUnsupported claim rate: {unsupported_rate:.1%}"
    )

    print(
        f"Judge human-needed rate: {human_rate:.1%}"
    )

    print(
        f"\nSaved judgments: {OUTPUT}"
    )


if __name__ == "__main__":
    main()