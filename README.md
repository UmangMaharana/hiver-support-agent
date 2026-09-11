# Tesco Customer Support Agent

A small, evaluation-first customer-support agent built from the **Kaggle Customer Support on Twitter** dataset.

The system takes an inbound customer message and:

1. classifies it into a frozen 13-intent taxonomy,
2. retrieves similar historical Tesco support cases and their resolution guidance,
3. drafts a concise reply with Gemini,
4. decides whether the case should be escalated to a human, and
5. evaluates the full pipeline on a 250-example hand-labelled golden set plus targeted human and LLM-judge evaluations.

> **Main finding:** the system is strongest at response generation and escalation, while **intent ambiguity and lexical retrieval** remain the main bottlenecks.

---

## 1. Project overview

This project is intentionally built as a small, reproducible support-agent pipeline rather than a large end-to-end ML system.

```text
Raw Twitter dataset
        │
        ▼
Conversation graph + Tesco filtering
        │
        ▼
Clean conversation cases
        │
        ├──────────────► Golden benchmark (250 hand-labelled)
        │
        └──────────────► Weakly-labelled silver set (3,073)
                                  │
                                  ▼
                       TF-IDF + Logistic Regression
                                  │
                                  ▼
                       Predicted support intent
                                  │
                                  ▼
                    TF-IDF historical retrieval
                                  │
                                  ▼
                     Gemini reply + escalation
                                  │
                                  ▼
                         Evaluation harness
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
             Intent          Escalation       Reply quality
                                             (LLM judge)
```

The system is designed to **draft safely**, not pretend to have live Tesco systems. Historical replies are treated as **historical resolution guidance**, not as proof of current Tesco policy, pricing, availability, or account state.

---
## 1A. Fast headline-result reproduction

A reviewer can reproduce the reported evaluation metrics without rebuilding the
~3M-row raw Twitter dataset or making new Gemini calls.

The repository should contain these small evaluation artifacts:

- `data/golden/tesco_golden_250_labeled.csv`
- `data/baselines/baseline_predictions.csv`
- `data/generation/reply_predictions.csv`
- `data/retrieval/retrieval_human_eval_150.csv`
- `data/escalation/escalation_human_eval_50.csv`
- `data/evaluation/reply_judgments.csv`

Then, after installing the project:

```bash
python -m pip install -e .
python -m src.evaluation.run_full_eval
```

This recomputes the headline intent, retrieval, escalation, generation-coverage,
and LLM-judge aggregates from the frozen evaluation artifacts. It is the
recommended **under-15-minute headline reproduction path**.

The full data rebuild (`build_graph` through generation/judging) is intentionally
separate: it requires the raw Kaggle dataset and Gemini API access and is not
claimed to fit the 15-minute reproduction target.

---

## 2. Dataset and scope

Source dataset:

**Customer Support on Twitter** — the Kaggle dataset containing customer-support conversations from multiple brands.

Tesco was selected as the project brand after comparing brand-level conversation volume. The explored Tesco subset contained:

- **38,573 Tesco-authored tweets**
- **16,722 connected conversation components**

After conversation cleaning and filtering, the project produced:

- **15,678 clean Tesco conversation cases**
- **250 hand-labelled golden examples**
- **3,073 weakly-labelled silver training examples**

The raw Kaggle CSV is intentionally **not committed to this repository**.

### Conversation construction

Tweets are represented as a conversation graph using the dataset's reply relationships. The Tesco case builder then keeps conversations satisfying the following constraints:

- exactly one distinct non-Tesco customer author,
- at least one customer turn,
- at least one Tesco turn,
- between **2 and 30 turns**.

This produces a cleaner corpus for support-case retrieval and avoids mixing multiple customers into a single case.

---

## 3. Frozen intent taxonomy

The classifier uses 13 intents. The taxonomy was frozen before the final golden-set evaluation.

| Intent | Definition |
|---|---|
| `delivery_issue` | Problems with an existing delivery being late, missing, damaged, or otherwise not delivered as expected. |
| `delivery_slots` | Questions or problems about booking, changing, or availability of delivery slots. |
| `online_website_issue` | Technical problems with Tesco website, app, online shopping systems, or related digital functionality. |
| `online_order_issue` | Problems with an existing online order, including missing/wrong items, cancellation, or order processing. |
| `click_collect` | Questions or problems specifically involving Click & Collect orders or collection. |
| `product_availability` | Whether/where/when a product is stocked or back in stock. |
| `product_quality_safety` | Spoiled, expired, contaminated, damaged, defective, unsafe, or poor-quality products. |
| `pricing_offers` | Incorrect prices, discounts, promotions, price labels, or offers. |
| `refund_return_compensation` | Refunds, returns, reimbursement, or compensation. |
| `store_service_complaint` | Staff, store-service, store-experience, facilities, checkout, or in-store-service complaints. |
| `product_information` | Ingredients, allergens, dietary information, or other product details. |
| `clubcard_account` | Clubcard, account access, vouchers, or account-related issues. |
| `other` | Non-support/social messages, praise, or unclear requests. |

---

## 4. Training data and leakage control

The classifier is trained on a **silver set** created with conservative weak supervision.

The silver-labeling process uses:

- ordered, first-match-wins regex rules for the intents,
- unmatched examples are dropped rather than forced into a class,
- exact-duplicate removal after text normalization,
- exclusion of the **250-example golden set** and a separate **100-example taxonomy-review set** before silver labeling.

The final silver set contains **3,073 examples**.

The golden set is therefore held out from the classifier's training pool and is used only for evaluation and downstream testing.

---

## 5. Model choices

### Intent classifier

Three simple baselines were evaluated on the 250-example golden set:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 21.6% | 2.7% |
| **TF-IDF + Logistic Regression** | **59.6%** | **59.8%** |
| TF-IDF + Linear SVM | 56.0% | 56.4% |

**TF-IDF + Logistic Regression** is wired into the downstream pipeline because it was the strongest of the tested baselines.

The result is useful but far from solved: **101/250 (40.4%) golden examples were misclassified**.

The largest confusion patterns are concentrated around ambiguous language, especially:

- `other` → `product_quality_safety`
- `other` → `store_service_complaint`
- `store_service_complaint` → `product_quality_safety`

This is the main classification bottleneck.

### Retrieval

The production retriever is deliberately simple:

- TF-IDF word n-grams,
- cosine similarity,
- historical Tesco cases,
- top-3 cases passed to generation.

A separate historical-resolution corpus extracts the most recent Tesco response containing an operational action or next step where possible, otherwise the final Tesco response. These are referred to throughout the project as **historical resolution guidance**.

They are **not treated as current Tesco policy**.

### Generation

Gemini is used to generate:

```json
{
  "reply": "...",
  "escalate": true,
  "escalation_reason": "..."
}
```

Structured JSON output is enforced through the Gemini response schema.

The prompt explicitly instructs the model to:

- treat the customer's actual message as the source of truth,
- use historical cases as examples rather than current policy,
- avoid inventing account access, live inventory, current prices, promotions, or internal actions,
- avoid promising refunds, replacements, compensation, investigations, or other actions the system cannot actually perform,
- escalate when human investigation, account-specific information, live/internal systems, sensitive handling, or unavailable actions are required,
- remain concise and natural.

---

## 6. Evaluation

The project uses several evaluation layers because a single similarity or accuracy number can be misleading.

### Intent classification — golden set

**n = 250**

- Majority: 21.6% accuracy / 2.7% macro-F1
- Logistic Regression: 59.6% accuracy / 59.8% macro-F1
- Linear SVM: 56.0% accuracy / 56.4% macro-F1
- Misclassified: 40.4%

### Retrieval

Automated TF-IDF screening was run over **250 golden queries × top-5**.

The automated retriever produced a near-100% non-zero similarity match rate. This is deliberately **not** treated as evidence that retrieval is good: lexical overlap only shows that some text is related.

Human usefulness evaluation was performed separately on a stratified sample of **50 queries** with top-3 retrieval, producing **153 scored pairs**.

#### Pair-level human scores

| Metric | Result |
|---|---:|
| Mean score (0–3) | 1.889 |
| Direct (score = 3) | 39.2% |
| Useful or better (score ≥ 2) | 64.1% |
| Irrelevant (score = 0) | 14.4% |

#### Query-level human scores

| Metric | Result |
|---|---:|
| Direct@1 | 41.2% |
| Useful@1 | 64.7% |
| Direct@3 | 72.5% |
| Useful@3 | 92.2% |

The important distinction is:

> A near-100% non-zero similarity rate does **not** mean the system found a useful precedent.

Only **41.2%** of queries had a directly useful precedent at rank 1, while allowing the best of the top 3 increased Direct@3 to **72.5%**.

This is the project's clearest misleading headline number.

### Escalation

A separate **50-example human escalation benchmark** was stratified across the 13 intents.

Human escalation rate: **70%**

| System | Accuracy | Precision | Recall | F1 | Escalation rate |
|---|---:|---:|---:|---:|---:|
| Gemini generation-time decision | 84.0% | 82.9% | 97.1% | 89.5% | 82% |
| Deterministic rule baseline | 52.0% | 92.3% | 34.3% | 50.0% | 26% |

The rule baseline is precise but misses too many cases requiring intervention. Gemini is much better at recall, at the cost of some over-escalation.

**Important evaluation caveat:** the human escalation annotation was not blinded to the model/rule outputs, so these results should be treated as directional rather than as a fully independent gold standard.

### Generation reliability

Across all 250 golden examples:

- Reply coverage: **100%**
- Valid structured escalation output: **100%**

### Reply-quality LLM judge

All 250 generated replies were scored by a Gemini-based judge using a 1–5 rubric.

**This is an LLM-judge evaluation, not a human quality evaluation.**

| Dimension | Mean |
|---|---:|
| Intent alignment | 4.944 / 5 |
| Helpfulness | 4.832 / 5 |
| Grounding | 4.896 / 5 |
| Safety / factuality | 4.956 / 5 |
| Escalation appropriateness | 4.956 / 5 |
| **Overall quality** | **4.868 / 5** |

Additional flags:

- Unsupported-claim rate: **1.6%**
- Judge "needs human" rate: **76.4%**

The 250 generated replies are a frozen evaluation artifact. **GOLD_001–GOLD_065
were generated under an earlier prompt revision; GOLD_066 onward used the improved
prompt shown in `src/generation/build_prompt.py`.** Therefore the 4.868/5 judge
result is a result for the frozen mixed-version artifact, not a clean measurement
of the final prompt alone.

The high judge scores should not be interpreted as proof of autonomous resolution. A substantial portion of high-quality outputs are high-quality because the system appropriately defers to a human.

### Human audit of the LLM judge

To test whether the LLM judge tracks human judgment, a blinded audit was run on **30 generated replies**, with at least two examples per intent plus four additional sampled cases. The human evaluator saw only the customer message and generated reply, not the judge scores, predicted intent, or model escalation decision. The human used the same six 1–5 dimensions as the LLM judge.

| Dimension | Human mean | LLM-judge mean | MAE | Exact agreement | Within ±1 |
|---|---:|---:|---:|---:|---:|
| Intent alignment | 4.00 | 5.00 | 1.00 | 43.3% | 70.0% |
| Helpfulness | 3.57 | 4.97 | 1.40 | 16.7% | 53.3% |
| Grounding | 3.87 | 5.00 | 1.13 | 26.7% | 63.3% |
| Safety / factuality | 3.57 | 5.00 | 1.43 | 26.7% | 60.0% |
| Escalation appropriateness | 3.93 | 5.00 | 1.07 | 33.3% | 66.7% |
| **Overall quality** | **4.10** | **5.00** | **0.90** | **26.7%** | **83.3%** |

The key calibration finding is that the judge was **systematically more generous** on this sample: it assigned an overall quality of 5 to all 30 replies, while the human scores ranged from 3 to 5. Overall exact agreement was only **26.7% (8/30)**, although **83.3% (25/30)** were within one point. Because the judge's overall scores were constant at 5, rank correlation is not meaningful for that dimension.

This means the 4.868/5 automated quality score should be treated as a **useful but poorly calibrated automated signal**, not as a direct estimate of human-perceived quality. The next iteration should use a more independent judge and a larger, doubly annotated sample.

There is also a methodological limitation: generation and judging use the same default Gemini model family, so this is not independent external grading.

---

## 7. Failure analysis

The evaluation shows a fairly clear pattern: **classification ambiguity and lexical retrieval are the main weaknesses; generation is comparatively strong but inherits upstream mistakes.**

### 1. `other` → `product_quality_safety`

This was the largest single confusion pair, with **18 cases**.

Example: `GOLD_002`

> “Since when have you started cutting crusts of children's sandwiches”

The message is better understood as a policy/commentary question, but the classifier interpreted it as a product-quality complaint. The downstream reply then asked for product/store details that were not needed.

### 2. `other` → `store_service_complaint`

**8 cases** fell into this confusion pair. The available failure artifact records the aggregate count but does not provide a specific representative case, so no case is invented here.

### 3. `store_service_complaint` → `product_quality_safety`

Another **8 cases**. Again, the available failure artifact provides the aggregate count but not a specific quoted case.

### 4. Lexical retrieval failure

`GOLD_003`:

> “estimate i've spent £78k with you...”

Human retrieval scoring gave this query a **0.0** example score. Numeric, highly specific, or account-specific language is difficult for pure lexical matching because the useful historical resolution may use very different wording.

Other weak examples include specific limits, named stores, and brand/product comparisons.

### 5. Rule-baseline under-escalation

`GOLD_231` describes leaking milk pints that damaged a customer's car.

The deterministic keyword policy failed to escalate it, while the human benchmark marked it as requiring human intervention.

This illustrates the limitation of a narrow regex policy: it can miss unusual but consequential cases even when the underlying need for investigation is clear.

### 6. Gemini over-escalation

`GOLD_085` asks about the discontinuation of 5p bags.

Gemini escalated it, while the human benchmark judged that a human was not required. The model is therefore conservative in some straightforward informational cases.

### 7. Unsupported current-policy claim

`GOLD_060` is a Clubcard example where the generated answer stated that Boost vouchers are “generally valid for 6 months”.

The claim was not grounded in the retrieved historical evidence. The judge gave the response **2/5**.

This is a useful reminder that a correct intent prediction does not guarantee a safe factual answer.

### 8. Wrong intent cascading downstream

`GOLD_028` was:

```text
Gold: product_availability
Predicted: online_order_issue
```

The wrong intent changed the downstream framing and retrieval, producing a reply that asked for personal details to investigate instead of addressing a general product-availability question.

This is the clearest example of why classification quality matters even when the generation model itself is strong.

---

## 8. Engineering roadmap

If this system were continued for another week, the highest-value changes would be:

### 1. Hybrid retrieval + reranking

**Problem:** TF-IDF misses semantically related but lexically different cases.

**Change:** combine lexical retrieval with embedding-based candidates, then rerank using intent compatibility, semantic similarity, and resolution quality.

**Expected benefit:** improve specific/numeric/account-specific retrieval and reduce irrelevant historical precedents.

**Measure:** Direct@1, Direct@3, Useful@1, Useful@3 on the same blinded human retrieval benchmark.

### 2. Taxonomy refinement / hierarchical classification

**Problem:** many errors come from genuinely overlapping categories rather than simple vocabulary misses.

**Change:** introduce hierarchical routing, for example support-vs-social first, then operational category; explicitly model common boundaries such as product quality vs store service.

**Expected benefit:** fewer `other`/quality/store-service confusions.

**Measure:** macro-F1 plus per-intent recall and confusion matrices on a refreshed golden set.

### 3. Capability-aware escalation

**Problem:** Gemini catches most human-needed cases but over-escalates some answerable informational questions.

**Change:** make capabilities explicit as structured state and separate “needs current/internal information” from “needs a human action”.

**Expected benefit:** preserve high recall while reducing unnecessary escalation.

**Measure:** human-annotated precision/recall/F1 on a larger blinded escalation set.

### 4. Stronger factuality/current-policy safeguards

**Problem:** historical responses can contain outdated policies or unsupported claims.

**Change:** require evidence-backed claims for policy/price/availability statements and use a dedicated abstain/verify path when current information is unavailable.

**Expected benefit:** lower unsupported-claim rate.

**Measure:** unsupported-claim rate plus human factuality ratings on adversarial current-policy examples.

### 5. Larger blinded human evaluation

**Problem:** the current escalation benchmark is only n=50 and is not blinded.

**Change:** expand the sample and blind annotators to model/rule outputs; ideally use multiple annotators.

**Expected benefit:** more reliable estimates of escalation quality and inter-annotator agreement.

**Measure:** precision, recall, F1, agreement, and annotator agreement on the expanded benchmark.

---

## 9. Reproducing the pipeline

### Requirements

- Python **3.11 or 3.12**
- Kaggle Customer Support on Twitter dataset
- Gemini API key for generation and LLM-judge stages

Install the package in editable mode:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
# JUDGE_MODEL=gemini-2.5-flash
```

Do not commit `.env` or API keys.

### Pipeline commands

Run the stages in order.

#### 1. Build the tweet database

```bash
python -m src.ingestion.build_graph
```

This imports the raw `twcs.csv` into the processed SQLite database.

#### 2. Build clean Tesco conversation cases

```bash
python -m src.ingestion.build_tesco_cases
```

#### 3. Build the weakly-labelled silver set

```bash
python -m src.baselines.build_silver_set
```

#### 4. Train/evaluate intent baselines

```bash
python -m src.baselines.train_baselines
```

Optional analysis:

```bash
python -m src.baselines.analyze_baselines
```

#### 5. Build the leakage-safe retrieval corpus

```bash
python -m src.retrieval.build_corpus
```

#### 6. Extract historical resolution guidance

```bash
python -m src.retrieval.extract_resolutions
```

#### 7. Run automated retrieval evaluation

```bash
python -m src.retrieval.evaluate_retrieval
```

#### 8. Build and annotate the retrieval human-evaluation sample

```bash
python -m src.retrieval.build_retrieval_human_eval
python -m src.retrieval.annotate_retrieval_human_eval
python -m src.retrieval.evaluate_human_retrieval
```

The annotation command is interactive.

#### 9. Generate replies and escalation decisions

```bash
python -m src.generation.generate_replies
```

The generation script is resume-safe and saves successful results incrementally.

#### 10. Evaluate the deterministic escalation baseline

```bash
python -m src.escalation.evaluate_policy
```

#### 11. Build and annotate the escalation human-evaluation sample

```bash
python -m src.escalation.build_escalation_human_eval
python -m src.escalation.annotate_escalation_human_eval
python -m src.escalation.evaluate_human_escalation
```

The annotation command is interactive.

#### 12. Judge generated replies

```bash
python -m src.evaluation.judge_replies
```

#### 13. Run the full evaluation harness

```bash
python -m src.evaluation.run_full_eval
```

The final summary is written under:

```text
data/evaluation/full_eval_summary.json
data/evaluation/full_eval_summary.csv
```

Failure analysis can be generated with:

```bash
python -m src.evaluation.analyze_failures
python -m src.evaluation.build_failure_report
python -m src.evaluation.build_evidence_table
```

---

## 10. Repository layout

```text
.
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── ingestion/
│   │   ├── build_graph.py
│   │   └── build_tesco_cases.py
│   │
│   ├── taxonomy/
│   │   └── taxonomy.py
│   │
│   ├── baselines/
│   │   ├── build_silver_set.py
│   │   ├── train_baselines.py
│   │   └── analyze_baselines.py
│   │
│   ├── retrieval/
│   │   ├── build_corpus.py
│   │   ├── tfidf_retriever.py
│   │   ├── extract_resolutions.py
│   │   ├── evaluate_retrieval.py
│   │   ├── build_retrieval_human_eval.py
│   │   ├── annotate_retrieval_human_eval.py
│   │   └── evaluate_human_retrieval.py
│   │
│   ├── generation/
│   │   ├── build_prompt.py
│   │   └── generate_replies.py
│   │
│   ├── escalation/
│   │   ├── escalation_policy.py
│   │   ├── evaluate_policy.py
│   │   ├── build_escalation_human_eval.py
│   │   ├── annotate_escalation_human_eval.py
│   │   └── evaluate_human_escalation.py
│   │
│   └── evaluation/
│       ├── judge_replies.py
│       ├── run_full_eval.py
│       ├── analyze_failures.py
│       ├── build_failure_report.py
│       └── build_evidence_table.py
│
├── data/
│   ├── golden/
│   ├── retrieval/
│   ├── escalation/
│   ├── generation/
│   └── evaluation/
│
└── tests/
```

Generated raw/derived data and secrets are excluded from Git by default. **Required small evaluation artifacts are explicitly allow-listed**: the 250-row golden set and the 30-row human-vs-LLM judge audit should be committed with the submission.

The concise submission report is [`REPORT.md`](REPORT.md). The human-vs-LLM audit artifact is `data/evaluation/reply_human_vs_llm_30_complete.csv`.

---

## 11. Key design decisions

A few decisions materially shaped the system:

| Decision | Why |
|---|---|
| Logistic Regression over majority/SVM | Best tested golden-set baseline |
| TF-IDF retrieval | Simple, cheap, and sufficient for a first implementation |
| Human retrieval evaluation | Cosine similarity alone does not measure resolution usefulness |
| Golden + taxonomy-review exclusion before silver labeling | Stronger leakage control |
| First-match-wins weak supervision | Precision-first rather than forcing noisy labels |
| Structured Gemini JSON | Reliable downstream parsing |
| Historical responses treated as guidance | Prevent historical text from becoming fake current policy |
| Deterministic escalation baseline | Provides an interpretable comparison point |
| Stratified human subsets | Avoids common intents dominating tiny human benchmarks |
| Top-3 retrieval for generation | More useful precedents than top-1 without flooding the prompt |
| Single-customer conversation filtering | Cleaner support cases at the cost of excluding some data |

---

## 12. Limitations

This is a scoped take-home project, not a production-ready support system.

The most important limitations are:

1. **Classification remains weak.** A 59.8% macro-F1 baseline leaves substantial downstream error.
2. **Human escalation evaluation is small (n=50).** One example can materially change recall/precision.
3. **Escalation annotation was not blinded.** Model outputs may have influenced human judgments.
4. **Generation and judging use the same default Gemini model family.** The high LLM-judge scores therefore are not independent quality certification.
5. **Judge retrieval differs from production retrieval.** The judge re-derives retrieval context using a different TF-IDF configuration rather than always receiving the exact evidence shown to the generator.
6. **Historical resolutions can become stale.** Prompt constraints reduce this risk but cannot guarantee factual correctness; the 1.6% unsupported-claim rate demonstrates that the safeguard is not perfect.
7. **No live Tesco systems are connected.** The agent cannot actually inspect an account, current store inventory, current promotions, orders, or internal support systems.
8. **Human-vs-judge reply-quality audit is small (n=30) and single-annotator.** It is useful calibration evidence, but not a statistically strong estimate of judge reliability; the judge was systematically more generous on this sample.

---

## 13. What I would improve first

If this were moving beyond the take-home:

**First:** fix retrieval and classification together.

A better classifier reduces bad retrieval framing, while semantic retrieval reduces the damage caused by lexical mismatch. These two components account for the clearest measured weaknesses.

**Second:** make capabilities explicit.

The model should know exactly which actions/data are available and turn unavailable operations into controlled escalation rather than guessing.

**Third:** replace the current small human benchmarks with a blinded, multi-annotator evaluation.

That would make escalation and retrieval claims much more defensible.

---

## 14. Final evaluation snapshot

| Component | Headline result |
|---|---|
| Intent classification | 59.6% accuracy / 59.8% macro-F1 |
| Intent errors | 40.4% of golden examples |
| Retrieval Direct@1 | 41.2% |
| Retrieval Direct@3 | 72.5% |
| Retrieval Useful@3 | 92.2% |
| Gemini escalation F1 | 89.5% on n=50 |
| Rule escalation F1 | 50.0% on n=50 |
| Reply coverage | 100% |
| Valid escalation JSON | 100% |
| LLM-judge overall quality | 4.868 / 5 |
| LLM-judge vs human overall: exact agreement / within ±1 | 26.7% / 83.3% (n=30) |
| Unsupported claims | 1.6% |

The headline is not “the agent solves customer support.”

It is that **the response-generation layer is already strong enough to be useful, but its ceiling is currently set by upstream intent ambiguity, retrieval quality, and the absence of live/current support capabilities.**
