# Final repo checklist

1. Copy/extract this patch into the repository root.
2. Ensure the required 250-row golden set is present locally at:
   `data/golden/tesco_golden_250_labeled.csv`
3. Because the project ignores generated data by default, force-add the small required evaluation artifacts and golden set:
   `git add -f data/golden/tesco_golden_250_labeled.csv data/evaluation/reply_human_eval_30.csv data/evaluation/reply_human_vs_llm_30_complete.csv`
4. Stage the updated docs:
   `git add README.md REPORT.md REPO_CHECKLIST.md`
5. Verify no secret or raw dataset is staged:
   `git status --short`
   `git diff --cached --name-only`
6. Run the lightweight checks:
   `python -m compileall -q src`
   `pytest -q`
7. Commit and push:
   `git commit -m "docs: finalize evaluation report and judge audit"`
   `git push origin main`

## Why the golden set matters

The assignment explicitly requires a 150–250 example hand-labelled golden set. It is small enough to commit and should not remain hidden behind the broad `data/` ignore rule. Do **not** commit the ~493 MB raw Twitter dataset, SQLite corpus, `.env`, or API keys.

## Optional small artifacts

If already present locally, these are also useful to commit for reproducibility/evidence:
- `data/evaluation/full_eval_summary.json`
- `data/evaluation/full_eval_summary.csv`
- `data/evaluation/failure_analysis.csv`
- `data/evaluation/failure_analysis_report.md`
- `data/evaluation/report_evidence.csv`
- `data/retrieval/retrieval_human_eval_150.csv`
- `data/escalation/escalation_human_eval_50.csv`

Keep the raw dataset, processed SQLite database, model caches, and secrets ignored.
