from pathlib import Path
import re

import pandas as pd


CASES_PATH = Path("data/processed/tesco_cases.parquet")
GOLDEN_PATH = Path("data/golden/tesco_golden_250_labeled.csv")
REVIEW_PATH = Path("data/golden/taxonomy_review_100.csv")
OUTPUT_PATH = Path("data/processed/tesco_silver_set.parquet")

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# High-precision keyword rules
#
# The goal is conservative weak supervision:
# - strong signals -> assign an intent
# - ambiguous messages -> leave unlabeled
# - first matching rule wins
# ---------------------------------------------------------------------------

RULES = [
    (
        "online_website_issue",
        [
            r"\bwebsite\b.*\b(?:down|crash|crashed|broken|error|issue|problem|won't|wont|can't|cant|unable)\b",
            r"\b(?:website|site)\b.*\b(?:checkout|refresh|load|login|access|browse|search|error|crash)\b",
            r"\bapp\b.*\b(?:down|crash|crashed|broken|error|issue|problem|won't|wont|can't|cant|unable)\b",
            r"\b(?:app|application)\b.*\b(?:checkout|refresh|load|login|access|browse|search|error|crash)\b",
            r"\bonline (?:grocery )?(?:shop|shopping|checkout)\b.*\b(?:down|error|broken|crash|problem|issue|stuck)\b",
            r"\b(?:grocery )?(?:website|site|app)\b.*\b(?:down|error|broken|crash|problem|issue|stuck)\b",
            r"\bpage\b.*\b(?:crash|error|broken|won't|wont|can't|cant|refresh|load|stuck)\b",
            r"\bsearch\b.*\b(?:website|app|online)\b.*\b(?:problem|issue|rubbish|wrong|broken)\b",
            r"\b(?:can't|cant|cannot|unable to)\b.*\b(?:login|log in|sign in|checkout)\b",
        ],
    ),

    (
        "delivery_slots",
        [
            r"\bdelivery slot\b",
            r"\bdelivery slots\b",
            r"\bno slots\b",
            r"\bno delivery slots\b",
            r"\bbook(?:ing)? (?:a|my|the)?\s*delivery slot\b",
            r"\bchange (?:my )?delivery slot\b",
            r"\bavailable delivery slots\b",
            r"\bdelivery slot availability\b",
            r"\bslot availability\b",
            r"\bdelivery saver\b.*\b(?:slot|priority|slots)\b",
        ],
    ),

    (
        "click_collect",
        [
            r"\bclick\s*(?:&|and)\s*collect\b",
            r"\bclick ?collect\b",
            r"\bclick\s*(?:&|and)\s*collect order\b",
            r"\bcollection (?:from|at) (?:store|shop)\b",
        ],
    ),

    (
        "clubcard_account",
        [
            r"\bclubcard\b",
            r"\bclub card\b",
            r"\bclubcard voucher\b",
            r"\bclubcard points\b",
            r"\bclubcard number\b",
            r"\bmy ?account\b.*\b(?:login|access|error|problem|issue|details)\b",
            r"\baccount (?:access|login|locked|problem|issue)\b",
            r"\baccount\b.*\b(?:password|login|details)\b",
            r"\bpay\+\b",
            r"\bpay at pump\b",
            r"\bpayatpump\b",
        ],
    ),

    (
        "refund_return_compensation",
        [
            r"\b(?:can|could|may|would|do|does|is|am)\b.*\brefund\b",
            r"\b(?:need|want|looking for|requesting)\b.*\brefund\b",
            r"\brefund\b.*\b(?:please|can|could|how|where|when)\b",
            r"\b(?:return|exchange)\b.*\b(?:item|product|goods|order|purchase)\b",
            r"\b(?:money back|reimbursement|compensation)\b",
            r"\brefunded\b",
            r"\bget (?:a )?refund\b",
            r"\bwant (?:a )?refund\b",
            r"\bcompensation\b",
            r"\breimburse(?:ment|d)?\b",
            r"\breturn\b.*\b(?:item|product|goods|order)\b",
            r"\breturn(?:ed|ing)?\b.*\b(?:item|product|goods|order)\b",
        ],
    ),

    (
        "product_quality_safety",
        [
            r"\bmou?ld(?:y|ed)?\b",
            r"\bexpired\b",
            r"\bout of date\b",
            r"\buse by date\b",
            r"\bbest before\b",
            r"\bforeign object\b",
            r"\bstone in\b",
            r"\binsect\b",
            r"\bbug\b",
            r"\brat\b",
            r"\bsnail\b",
            r"\bcontaminat(?:ed|ion)\b",
            r"\bspoilt\b",
            r"\bspoiled\b",
            r"\bstale\b",
            r"\bdefective\b",
            r"\bdamaged (?:product|food|item|pack|goods)\b",
            r"\bpoor quality\b",
            r"\bquality issue\b",
            r"\bquality problem\b",
            r"\bcracked\b",
            r"\bunderfilled\b",
            r"\bunder filled\b",
            r"\bnot enough (?:in|inside)\b",
            r"\bmissing (?:pieces|contents)\b",
            r"\bwrong (?:flavour|flavor)\b",
            r"\bgristle\b",
            r"\btoo much waste\b",
            r"\ballergen\b",
            r"\ballergic reaction\b",
            r"\bunsafe\b",
            r"\bnearly (?:broke|lost|cut)\b.*\b(?:tooth|finger|hand)\b",
            r"\b(?:bone|stone|insect|bug|snail)\b.*\b(?:food|bread|meat|product)\b",
            r"\bproduct\b.*\b(?:broke|broken|faulty|defective|damaged)\b",
            r"\bfood\b.*\b(?:broke|broken|faulty|defective|damaged)\b",
            r"\bgone off\b",
            r"\bgone bad\b",
            r"\bspoiled\b",
            r"\bspoilt\b",
            r"\bout of date\b",
        ],
    ),

    (
        "pricing_offers",
        [
            r"\bprice match\b",
            r"\bprice matching\b",
            r"\bdiscount\b",
            r"\bdiscounted\b",
            r"\bpromotion\b",
            r"\bpromo\b",
            r"\bwrong price\b",
            r"\bincorrect price\b",
            r"\bprice(?:d|ing)?\b.*\b(?:wrong|incorrect|mistake)\b",
            r"\bcharged (?:too much|more)\b",
            r"\bcharged more\b",
            r"\bprice difference\b",
            r"\bprice increase\b",
            r"\bmisleading price\b",
            r"\bmisleading pricing\b",
            r"\bcheaper online\b",
            r"\bmore expensive\b",
            r"\bdiscount\b.*\b(?:applied|missing|wrong|failed)\b",
            r"\b(?:price|priced|pricing|cost|charged|charge)\b.*\b(?:wrong|incorrect|higher|more|too much|different)\b",
            r"\b(?:wrong|incorrect|higher|more|too much|different)\b.*\b(?:price|priced|pricing|cost|charged|charge)\b",
            r"\b(?:discount|promotion|promo|deal)\b.*\b(?:not working|doesn't work|doesnt work|won't work|wont work|missing|wrong|incorrect)\b",
            r"\b(?:offer|offers)\b.*\b(?:price|discount|voucher|checkout|working|work|valid|expired)\b",
            r"\b(?:price|prices)\b.*\b(?:offer|offers|discount|deal|promotion)\b",
        ],
    ),

    (
        "product_information",
        [
            r"\bwhat (?:are|is)\b.*\b(?:ingredients?|allergens?|calories?|nutrition)\b",
            r"\bwhat does\b.*\bcontain\b",
            r"\bdoes\b.*\bcontain\b.*\b(?:milk|nuts?|peanuts?|gluten|soya|soy|egg)\b",
            r"\bcontains?\b.*\b(?:milk|nuts?|peanuts?|gluten|soya|soy|egg)\b",
            r"\bingredients?\b.*\b(?:list|information|contain)\b",
            r"\ballergens?\b",
            r"\bcalorie(?:s)?\b.*\b(?:information|content|count)\b",
            r"\bnutrition(?:al)?\b.*\b(?:information|content)\b",
            r"\bdietary\b.*\b(?:information|requirements|suitability)\b",
            r"\bis this\b.*\b(?:vegan|vegetarian|gluten[- ]?free|dairy[- ]?free)\b",
            r"\bis .* suitable for\b.*\b(?:vegan|vegetarian|allerg|coeliac|celiac)\b",
            r"\brecipe\b.*\b(?:changed|change|different)\b",
            r"\bwhy (?:have you|did you)\b.*\brecipe\b",
            r"\bproduct (?:information|details)\b",
            r"\bhow much\b.*\b(?:weigh|weight)\b",
        ],
    ),

    (
        "product_availability",
        [
            r"\bwhere can i (?:buy|purchase|find)\b",
            r"\bwhere can we (?:buy|purchase|find)\b",
            r"\bwhich stores\b.*\b(?:stock|sell|have)\b",
            r"\bwhat stores\b.*\b(?:stock|sell|have)\b",
            r"\bavailable (?:in store|in-store)\b",
            r"\bavailable to purchase\b",
            r"\bavailable in\b.*\b(?:store|tesco)\b",
            r"\bout of stock\b.*\b(?:store|shop|shelf)\b",
            r"\brestock(?:ed|ing)?\b",
            r"\bback in stock\b",
            r"\bwill you be getting more\b",
            r"\bwill you be restocking\b",
            r"\bdiscontinued\b",
            r"\b(?:why|when|have you)\b.*\b(?:stopped selling|stop selling|no longer sell)\b.*\b(?:product|item|brand|sandwich|milk|bread|drink)\b",
            r"\b(?:discontinued|stopped selling|no longer sell)\b.*\b(?:product|item|brand|sandwich|milk|bread|drink)\b",
            r"\bwhere .* stock\b",
        ],
    ),

    (
        "delivery_issue",
        [
            r"\bdelivery (?:is|was|has been) late\b",
            r"\blate delivery\b",
            r"\bdelivery is \d+ .* late\b",
            r"\bdelivery .* overdue\b",
            r"\bdelivery .* no show\b",
            r"\bdelivery .* missing\b",
            r"\bdelivery .* hasn't arrived\b",
            r"\bdelivery .* has not arrived\b",
            r"\bdelivery .* didn't arrive\b",
            r"\bdelivery .* did not arrive\b",
            r"\bdelivery .* not arrived\b",
            r"\bdelivery .* marked as delivered\b",
            r"\bdelivery .* damaged\b",
            r"\bwhere is my delivery\b",
            r"\bwhere's my delivery\b",
            r"\bdelivery\b.*\b(?:late|delayed|missing|no show|not arrived)\b",
            r"\b(?:delivery|deliver)\b.*\b(?:poor|bad|terrible|awful|late|delayed|missing|failed|can't|cant|won't|wont)\b",
            r"\b(?:can't|cant|won't|wont|unable to)\b.*\bdeliver\b",
        ],
    ),

    (
        "online_order_issue",
        [
            r"\bonline order\b",
            r"\bordered online\b",
            r"\border online\b",
            r"\bmy order\b",
            r"\bthe order\b",
            r"\border .* cancelled\b",
            r"\border .* canceled\b",
            r"\border .* missing\b",
            r"\border .* incorrect\b",
            r"\border .* not arrived\b",
            r"\border .* hasn't arrived\b",
            r"\border .* has not arrived\b",
            r"\border .* delayed\b",
            r"\border .* despatched\b",
            r"\border .* dispatched\b",
            r"\border confirmation\b",
            r"\bcancel(?:l)? .* order\b",
            r"\bmissing items? .* order\b",
            r"\bsubstitut(?:e|ion|ed)\b",
        ],
    ),

    (
        "store_service_complaint",
        [
            r"\bpoor service\b",
            r"\bterrible service\b",
            r"\bdisgraceful service\b",
            r"\bdisgusting service\b",
            r"\bawful service\b",
            r"\bbad service\b",
            r"\bcustomer service\b.*\b(?:poor|bad|terrible|awful|disgusting|disgraceful|complaint)\b",
            r"\bcustomer care\b.*\b(?:poor|bad|terrible|awful|disgusting|disgraceful|complaint)\b",
            r"\bstaff\b.*\b(?:rude|attitude|service|treated|treatment)\b",
            r"\b(?:rude|bad) (?:staff|service)\b",
            r"\bbad attitude\b",
            r"\bstore\b.*\b(?:dirty|smell|smelly|unsafe|dangerous)\b",
            r"\bstore\b.*\b(?:complaint|complain)\b",
            r"\bcheckout\b.*\b(?:staff|service|queue|queuing)\b",
            r"\bqueues?\b.*\b(?:checkout|store|tesco)\b",
            r"\bparking\b.*\b(?:store|tesco)\b",
            r"\bfire hazard\b",
            r"\bstore\b.*\b(?:safety|hazard)\b",
        ],
    ),

    (
        "other",
        [
            r"^\s*thanks?\s*@?tesco\b",
            r"^\s*thank you\s*@?tesco\b",
            r"^\s*@?tesco\s+(?:thanks?|thank you)\b",
            r"^\s*@?tesco\s+(?:you're|you are)\s+the best\b",
            r"^\s*@?tesco\s+(?:great|awesome|brilliant|fantastic)\b",
            r"^\s*thank(?:s| you)\b.*\b(?:great|brilliant|awesome|fantastic|help)\b",
        ],
    ),
]


COMPILED_RULES = [
    (
        intent,
        [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns],
    )
    for intent, patterns in RULES
]


def classify(text: str) -> tuple[str | None, str | None]:
    """Return (intent, matched_pattern) using the first high-precision rule."""
    if not isinstance(text, str):
        return None, None

    text = text.strip()
    if not text:
        return None, None

    for intent, patterns in COMPILED_RULES:
        for pattern in patterns:
            if pattern.search(text):
                return intent, pattern.pattern

    return None, None


def normalize_text(text: str) -> str:
    """Normalize text for exact duplicate detection."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Loading Tesco cases...")
    cases = pd.read_parquet(CASES_PATH)

    print(f"Cases loaded: {len(cases):,}")

    # -----------------------------------------------------------------------
    # Keep the golden benchmark and taxonomy-review set completely separate.
    # -----------------------------------------------------------------------
    golden = pd.read_csv(GOLDEN_PATH)
    review = pd.read_csv(REVIEW_PATH)

    excluded_case_ids = set(golden["case_id"].astype(str))
    excluded_case_ids.update(review["case_id"].astype(str))

    cases["case_id"] = cases["case_id"].astype(str)

    eligible = cases[
        ~cases["case_id"].isin(excluded_case_ids)
    ].copy()

    print(f"Excluded benchmark/review cases: {len(excluded_case_ids):,}")
    print(f"Eligible cases: {len(eligible):,}")

    # -----------------------------------------------------------------------
    # Only classify the initial customer message.
    # -----------------------------------------------------------------------
    eligible = eligible[
        eligible["initial_customer_message"].notna()
        & eligible["initial_customer_message"].str.strip().ne("")
    ].copy()

    eligible["silver_intent"], eligible["matched_rule"] = zip(
        *eligible["initial_customer_message"].map(classify)
    )

    silver = eligible[
        eligible["silver_intent"].notna()
    ].copy()

    # -----------------------------------------------------------------------
    # Remove exact duplicate messages from the silver corpus.
    # This reduces the chance that repeated/template tweets dominate training.
    # -----------------------------------------------------------------------
    silver["normalized_text"] = silver["initial_customer_message"].map(
        normalize_text
    )

    before_dedup = len(silver)

    silver = silver[
        silver["normalized_text"].ne("")
    ].drop_duplicates(
        subset=["normalized_text"],
        keep="first",
    ).copy()

    print(f"Rule-labeled examples: {before_dedup:,}")
    print(f"After exact-text deduplication: {len(silver):,}")

    # Keep the output focused on useful training metadata.
    columns = [
        "case_id",
        "root_tweet_id",
        "customer_id",
        "created_at",
        "turn_count",
        "initial_customer_message",
        "silver_intent",
        "matched_rule",
    ]

    silver = silver[columns].sort_values("case_id")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved silver set to: {OUTPUT_PATH}")

    print("\nSilver label distribution:")
    print(
        silver["silver_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()