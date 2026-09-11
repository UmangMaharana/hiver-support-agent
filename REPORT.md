# Tesco AI Support Agent

## 1. Problem framing

The goal is to build a support agent for Tesco's Twitter support queue that can make three decisions for every inbound customer message:

1. classify the message into a small, data-derived intent taxonomy;
2. draft a reply grounded in how Tesco historically handled similar issues; and
3. decide whether the case can be auto-handled or should be escalated to a human, with a reason.

For this project, **good** means a response is relevant to the customer's actual request, grounded in historical evidence without inventing current policy, concise and safe, and conservative about cases requiring account access, live information, investigation, or human action.

I deliberately did **not** build a production support integration, live Tesco policy/inventory lookup, account/order access, payment processing, refund execution, or a fully autonomous resolution workflow. The system is a scoped, evaluable prototype: it drafts and triages; it does not pretend to have capabilities it does not have.

### Why Tesco?

I compared brands in the Customer Support on Twitter dataset and selected Tesco because it provided substantial multi-turn support data: **38,573 Tesco-authored tweets across 16,722 connected components** before case filtering. This gave enough volume and operational diversity to construct a focused support corpus.

### Architecture

```text
TWCS raw tweets
      |
      v
Conversation graph
      |
      v
Clean Tesco cases
      |
      +---------------------> Weak supervision -> Silver set
      |                                      |
      |                                      v
      |                              TF-IDF + Logistic Regression
      |                                      |
      v                                      |
250 hand-labelled golden set <---------------+
      |
      +--> TF-IDF historical-case retrieval
      |             |
      |             v
      |          top-3 cases
      |             |
      +-------------+
                    v
             Gemini 2.5 Flash
                    |
              +-----+-----+
              |           |
            reply     escalate + reason
              |
              v
       Evaluation harness
       (automated + human + LLM judge)
```

The customer message is explicitly the highest-priority source of truth. Retrieved conversations are treated as historical guidance, not proof of today's prices, availability, policies, or processes.

---

## 2. Data and evaluation design

The raw Kaggle dataset contains roughly 3M Twitter support tweets and includes tweet IDs, author IDs, timestamps, inbound/outbound indicators, text, and response relationships.

I constructed a conversation graph using `in_response_to_tweet_id`. Connected components were then filtered to produce coherent evaluation units:

- exactly one customer author;
- at least one Tesco turn;
- at least one customer turn;
- 2–30 total turns.

This produced **15,678 clean Tesco conversation cases**.

### Frozen intent taxonomy

The final taxonomy contains 13 intents:

| Intent | Definition |
|---|---|
| `delivery_issue` | Existing delivery is late, missing, damaged, wrong, or otherwise problematic |
| `delivery_slots` | Booking, changing, or availability of delivery slots |
| `online_website_issue` | Tesco website/app/online-shopping technical problems |
| `online_order_issue` | Existing online-order problems, including missing/wrong items or cancellation |
| `click_collect` | Click & Collect questions or problems |
| `product_availability` | Whether, where, or when a product is stocked/back in stock |
| `product_quality_safety` | Spoiled, expired, contaminated, damaged, defective, unsafe, or poor-quality products |
| `pricing_offers` | Incorrect prices, discounts, promotions, price labels, or offers |
| `refund_return_compensation` | Refunds, returns, reimbursement, or compensation |
| `store_service_complaint` | Staff, store service, facilities, checkout, and in-store experience complaints |
| `product_information` | Ingredients, allergens, dietary, and other product details |
| `clubcard_account` | Clubcard/account access, vouchers, and account-related issues |
| `other` | Non-support, social, praise, or unclear messages |

The taxonomy was frozen before final golden evaluation.

### Training and leakage control

A conservative rule-based weak-supervision process produced a **3,073-example silver set**. Rules used first-match-wins behavior; unmatched examples were discarded and exact duplicates removed.

The **250-example golden set** was hand-labelled and kept out of silver training. A separate 100-example taxonomy-review set was also excluded from the eligible silver pool before labeling. This was done before model fitting rather than merely hiding examples at train/test split time.

### Evaluation layers

The evaluation was intentionally split by what each metric can actually prove:

- **Intent:** 250-example hand-labelled golden set.
- **Retrieval:** automated TF-IDF screening plus a separate human usefulness benchmark of 50 stratified queries and 153 scored top-3 pairs.
- **Escalation:** 50 stratified human-labelled examples.
- **Generation reliability:** structured-output coverage on all 250 golden examples.
- **Reply quality:** Gemini LLM judge on all 250 generated replies.

The LLM judge is explicitly an **LLM evaluation, not a human evaluation**.

---

## 3. Results

### 3.1 Intent classification

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 21.6% | 2.7% |
| TF-IDF + Linear SVM | 56.0% | 56.4% |
| **TF-IDF + Logistic Regression** | **59.6%** | **59.8%** |

Logistic Regression was selected for the downstream pipeline because it was the strongest of the three baselines.

The result is useful but not solved: **101/250 golden examples (40.4%) were misclassified**. The errors concentrate around genuinely ambiguous boundaries, particularly social/`other` messages, store-service complaints, product-quality complaints, pricing, and availability.

### 3.2 Retrieval

Production retrieval uses TF-IDF word n-grams (1–2) and cosine similarity over the leakage-safe historical corpus.

Automated screening produced a **100% non-zero match rate** for the 250 golden queries (both top-1 and top-5 non-zero fields were 100%). This is deliberately not presented as retrieval quality.

Human evaluation tells a different story:

#### Human query-level results — 50 queries

| Metric | Result |
|---|---:|
| Direct@1 | **41.2%** |
| Useful@1 | **64.7%** |
| Direct@3 | **72.5%** |
| Useful@3 | **92.2%** |

#### Human pair-level results — 153 scored pairs

| Metric | Result |
|---|---:|
| Mean score (0–3) | 1.889 |
| Direct (score = 3) | 39.2% |
| Useful-or-better (score ≥ 2) | 64.1% |
| Irrelevant (score = 0) | 14.4% |

### What is misleading about my headline number?

**“100% of queries had a non-zero retrieval match” sounds excellent, but it only proves that TF-IDF found lexically related text. It does not prove that the historical case is useful for resolving the customer's issue.**

The human benchmark shows the distinction clearly: only **41.2%** of queries had a directly useful precedent at rank 1. Allowing three candidates raises Direct@3 to **72.5%**, which supports using top-3 retrieval, but even that should not be confused with verified correctness.

This is the main reason retrieval is treated as a bottleneck rather than as a solved component.

### 3.3 Escalation

Human annotation on 50 stratified examples marked **70%** as requiring human intervention.

| System | Accuracy | Precision | Recall | F1 | Escalation rate |
|---|---:|---:|---:|---:|---:|
| **Gemini generation-time decision** | **84.0%** | 82.9% | **97.1%** | **89.5%** | 82% |
| Rule baseline | 52.0% | **92.3%** | 34.3% | 50.0% | 26% |
| Human benchmark | — | — | — | — | 70% |

Gemini catches substantially more human-labelled escalation cases. The deterministic policy is precise when it fires but misses many cases involving account-specific access, live information, investigation, or unusual complaints.

The human annotation was not blinded to model outputs, so these results are directional rather than an independent production certification.

### 3.4 Generation reliability

| Metric | Result |
|---|---:|
| Reply coverage | **100% (250/250)** |
| Valid structured escalation output | **100%** |

The generation contract returns exactly the fields needed downstream: `reply`, `escalate`, and `escalation_reason`.

### 3.5 Reply quality — LLM judge

A Gemini-based judge evaluated all 250 replies using a 1–5 rubric.

| Dimension | Mean |
|---|---:|
| Intent alignment | 4.944 |
| Helpfulness | 4.832 |
| Grounding | 4.896 |
| Safety / factuality | 4.956 |
| Escalation appropriateness | 4.956 |
| **Overall quality** | **4.868** |
| Unsupported-claim rate | **1.6%** |
| Judge “needs human” rate | **76.4%** |

These are **LLM-judge scores, not human quality scores**. The same default Gemini model family was used for generation and judging, so the results should not be treated as independent certification.

The 76.4% human-needed flag is also important context: high reply quality often reflects a cautious response that appropriately defers to a human. It is not evidence that the system autonomously resolves 76% of support cases.

#### Human audit of the judge

To test whether the judge tracks human judgment, I ran a small blinded audit on **30 generated replies**, constructed as at least two examples per intent plus four additional sampled cases. The human evaluator saw the customer message and generated reply, but not the judge's scores, predicted intent, or escalation decision, and scored the same six dimensions on the same 1–5 rubric.

| Dimension | Human mean | LLM-judge mean | MAE | Exact agreement | Within ±1 |
|---|---:|---:|---:|---:|---:|
| Intent alignment | 4.00 | 5.00 | 1.00 | 43.3% | 70.0% |
| Helpfulness | 3.57 | 4.97 | 1.40 | 16.7% | 53.3% |
| Grounding | 3.87 | 5.00 | 1.13 | 26.7% | 63.3% |
| Safety / factuality | 3.57 | 5.00 | 1.43 | 26.7% | 60.0% |
| Escalation appropriateness | 3.93 | 5.00 | 1.07 | 33.3% | 66.7% |
| **Overall quality** | **4.10** | **5.00** | **0.90** | **26.7%** | **83.3%** |

The main finding is not that the judge is perfectly calibrated: on this sample it was **systematically more generous than the human**. It assigned an overall score of 5 to all 30 replies, while the human gave 3 to 5, with a mean of 4.10. Overall exact agreement was therefore only **26.7% (8/30)**, although **83.3% (25/30)** were within one point. A rank correlation is not meaningful here because the judge's overall scores were constant at 5 on all 30 audited replies.

This audit materially changes how the 4.868/5 headline should be read: it is useful as a **consistent automated quality signal**, but not as a calibrated estimate of human-perceived quality. The next iteration should use an independent judge and a larger, doubly-annotated sample.

---

## 4. Failure analysis

### 1. `other` → `product_quality_safety`

This was the largest individual intent confusion pair, with **18 cases**.

**GOLD_002:** “Since when have you started cutting crusts of children's sandwiches”

The classifier interpreted the message as a product-quality complaint rather than a general policy/commentary question. The downstream response consequently asked for product/store details that were unnecessary.

**Hypothesis:** retail vocabulary such as “sandwiches” and “cutting” provides stronger lexical evidence than the conversational intent. A support-vs-social first stage could reduce this boundary error.

### 2. Lexical retrieval failure

**GOLD_003:** “estimate i've spent £78k with you...”

The top retrieved historical example received a **0.0 human usefulness score**. Other weak cases included queries about wine-order limits, an Apple Pay limit, a brand nutrition comparison, and named-store stock.

**Hypothesis:** TF-IDF is strong when the customer's wording resembles historical wording, but weak on numeric, account-specific, named-entity, or unusual requests. Semantic retrieval plus reranking should help.

### 3. Rule-baseline under-escalation

**GOLD_231** involved leaking milk pints that damaged a customer's car. Human annotation marked it as requiring human intervention, but the deterministic rule baseline did not escalate.

Other rule false negatives included account-specific Delivery Saver eligibility, an unexpected Delivery Saver charge, allergy-related store handling, live chat failure, delivery scheduling, account access, and recurring short-date product issues.

**Hypothesis:** fixed keywords cannot represent the combination of customer impact, investigation requirement, account context, and unavailable agent capability.

### 4. Gemini over-escalation

**GOLD_085** concerned the discontinuation of 5p bags. Human annotation marked it as answerable without human intervention, while Gemini escalated.

Other false positives included casual social commentary, general availability suggestions, general menu questions, nutrition questions, and general pricing questions.

**Hypothesis:** the model sometimes equates uncertainty or negative sentiment with a need for human intervention. Escalation should instead be explicitly tied to unavailable capabilities or investigation requirements.

### 5. Unsupported current-policy claim

**GOLD_060** was correctly classified as `clubcard_account`, but the reply stated that Clubcard Boost vouchers are “generally valid for 6 months.” The judge treated this as an unsupported factual claim and rated the reply poorly.

**Hypothesis:** historical support replies can tempt the generator to convert past statements into present-tense policy. The prompt now explicitly treats retrieved responses as historical evidence rather than current policy, but this case shows that prompt-only safeguards are not sufficient.

### 6. Wrong intent causing a downstream cascade

**GOLD_028** was `product_availability` but predicted as `online_order_issue`. The generated reply then asked for personal information to investigate an issue that was actually a general product-availability question. The reply received a judge score of 1/5.

**Hypothesis:** classification errors are disproportionately costly because intent is upstream of retrieval and generation. Improving classification can therefore improve multiple downstream metrics simultaneously.

---

## 5. What I would do with one more week

### 1. Hybrid retrieval + reranking

**Problem:** TF-IDF retrieval misses semantically similar but lexically different cases.

**Change:** combine lexical retrieval with dense embeddings, then rerank the candidate set using a cross-encoder or LLM.

**Expected benefit:** improve Direct@1 and reduce irrelevant historical precedents.

**Measure:** repeat the same blinded 50-query retrieval benchmark, especially Direct@1 and Useful@1.

### 2. Hierarchical intent classification

**Problem:** most classification errors occur at ambiguous boundaries such as social `other`, store service, product quality, pricing, and availability.

**Change:** first classify support-vs-social/other, then classify the support domain.

**Expected benefit:** reduce boundary confusion without making the top-level taxonomy unnecessarily large.

**Measure:** accuracy/macro-F1 on a newly blinded golden set and per-intent confusion rates.

### 3. Capability-aware escalation

**Problem:** intent alone does not determine whether a human is needed.

**Change:** explicitly model capabilities such as account access, order lookup, live inventory, current pricing, refunds, and investigation.

**Expected benefit:** fewer false positives on answerable informational questions and fewer false negatives on cases requiring human-only actions.

**Measure:** expand the human escalation benchmark and report precision/recall/F1 plus false-negative count.

### 4. Current-policy and factuality safeguards

**Problem:** historical resolutions can become stale.

**Change:** add a “current fact required?” gate. Current policy, price, inventory, or account claims should require a live source; otherwise the agent should use cautious non-claiming language.

**Expected benefit:** reduce unsupported claims below the current 1.6%.

**Measure:** adversarial factuality set targeting dates, prices, limits, eligibility, and availability.

### 5. Larger blinded human evaluation

**Problem:** the current human escalation benchmark is only n=50 and is not blinded.

**Change:** evaluate a larger stratified sample with annotators blinded to model decisions. Also have humans score a sample of generated replies using the same reply-quality rubric as the LLM judge.

**Expected benefit:** independent estimates of escalation quality and evidence of whether the LLM judge tracks human judgments.

**Measure:** human-vs-judge agreement/correlation on the same replies, plus confidence intervals for the main human metrics.

---

## 6. Limitations and decision log

### Limitations

1. **Small human escalation sample (n=50).** The benchmark is useful for diagnosis but too small for tight statistical conclusions; a one-example change materially affects recall.
2. **Escalation annotation was not blinded.** Annotators saw model and rule decisions/reasons, creating anchoring risk.
3. **Generation and LLM judge use the same default Gemini model family.** This makes the judge a self-evaluation rather than an independent evaluator.
4. **The judge re-derived retrieval context rather than consuming exactly the generator's stored top-3 evidence.** Its grounding score is therefore not a strict replay of the generation-time evidence.
5. **Historical resolutions are not current policy.** Prompt safeguards reduce the risk but do not eliminate it; GOLD_060 demonstrates the remaining failure mode.
6. **Human-vs-judge reply-quality audit is small (n=30) and single-annotator.** The audit provides useful calibration evidence, but it is not a statistically strong estimate of judge reliability. On this sample the judge was systematically more generous, including an overall score of 5 on all 30 replies.

### Decision log

| # | Decision | Alternatives | Why | Consequence |
|---|---|---|---|---|
| 1 | Use TF-IDF + Logistic Regression | Majority, Linear SVM | Best golden-set baseline: 59.6% accuracy / 59.8% macro-F1 | 40.4% intent error remains |
| 2 | Use lexical TF-IDF retrieval | Embeddings, hybrid retrieval | Simple, transparent, low infrastructure | Weak on semantic/numeric/account-specific queries |
| 3 | Evaluate retrieval with humans | Cosine similarity alone | Similarity is not usefulness | Exposed the 100% vs 41.2% headline gap |
| 4 | Exclude golden + 100 review examples before silver labeling | Exclude only at train/test split | Stronger leakage control | Smaller eligible silver pool |
| 5 | First-match-wins weak supervision; unmatched dropped | Force labels | Precision-first labeling | 3,073 silver examples |
| 6 | Structured JSON generation | Free-text parsing | Reliable downstream contract | 100% valid escalation output |
| 7 | Treat historical replies as historical evidence | Allow model to reuse facts directly | Avoid stale policy claims | Unsupported claims still occur at 1.6% |
| 8 | Build deterministic escalation baseline | LLM-only evaluation | Cheap, interpretable comparator | Rule recall only 34.3% |
| 9 | Stratify human retrieval/escalation samples by intent | Pure random sampling | Preserve coverage across 13 intents | Only 3–4 examples per intent in small samples |
| 10 | Do not claim blinded escalation ground truth | Blind annotation | Existing annotation workflow exposed model outputs | Anchoring risk must be reported |
| 11 | Use same default Gemini family for generation and judging | Independent judge model | Practical scoped-project choice | Self-evaluation caveat |
| 12 | Keep judge retrieval separate from production retrieval | Replay exact generator context | Existing implementation choice | Grounding evaluation has an evidence mismatch |
| 13 | Use top-3 retrieval for generation | Top-1, top-5 | More useful precedents without excessive context | Better coverage than top-1, but weaker matches can enter context |
| 14 | Keep single-customer, 2–30-turn case filter | No filtering / broader bounds | Cleaner support units | Excludes some raw conversations |
| 15 | Audit the LLM judge with blinded human scoring | Report judge scores alone | Check calibration against human perception | Exposed generous judge bias; n=30 is still small |

---

## Conclusion

The system is strongest at **response generation and escalation**, while **intent ambiguity and lexical retrieval remain the main bottlenecks**.

The classifier establishes a meaningful improvement over trivial and simple baselines, but 40.4% intent error shows that the problem is not solved. Retrieval demonstrates an especially important lesson: a near-100% lexical-match rate is a misleading success metric when only 41.2% of first-ranked precedents are directly useful to a human evaluator.

Generation is comparatively strong under the LLM judge, with 4.868/5 overall quality and 1.6% unsupported claims, but those scores must be interpreted alongside the self-evaluation caveat and the 76.4% judge human-needed rate.

The prototype therefore succeeds less by pretending to be an autonomous support agent and more by demonstrating a measurable path toward one: **clean data construction, a frozen taxonomy, explicit leakage controls, simple baselines, human retrieval and escalation evaluation, constrained generation, and honest failure analysis.**

The highest-value next step is not another larger language model. It is improving the **classification → retrieval → capability-aware escalation chain** and validating those improvements with a larger, blinded human evaluation.
