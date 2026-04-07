# Decisions & Tradeoffs Log

Every major decision made during this project, with reasoning. Reference this when explaining choices in interviews.

---

## Project-Level Decisions

### D1: Why e-commerce product reviews?
**Decision:** Build in e-commerce domain using Amazon Reviews dataset.
**Alternatives considered:** Healthcare (clinical trials), fintech (fraud detection), job market intelligence.
**Why this won:** 
- Free, high-quality data (571M reviews from McAuley Lab UCSD)
- Clear business impact ($50B+/year in returns)
- Maps to target companies: Amazon, Walmart, Wayfair, Shopify
- Different domain from other 2 projects (finance, supply chain)
**Tradeoff:** Two of three projects are retail-adjacent. Accepted because tech stacks are completely different.

### D2: Why this specific tech stack?
**Decision:** PyTorch + QLoRA + LangGraph + MLflow + Evidently + FastAPI + Docker
**Why:** Cross-referenced against actual 2026 job postings from Capital One, Meta, Amazon, Microsoft, TikTok, Deloitte, Wayfair, Oracle. Every tool matched 5+ out of 8 companies.
**What we deliberately excluded:**
- TensorFlow — using both PyTorch AND TF looks unfocused. PyTorch has 55% production share in 2026.
- Kubernetes — overkill for a portfolio project. Docker Compose is sufficient.
- Kafka — no real streaming need. Batch processing is honest and realistic.
- Airflow — already demonstrated in Last-Mile project. DVC pipelines sufficient here.

### D3: Why Snowflake + PostgreSQL (both)?
**Decision:** Use Snowflake as data warehouse for raw reviews, PostgreSQL as application database.
**Why Snowflake:** Already on resume from Last-Mile + Airbnb projects. Reinforces expertise. Free 30-day trial with $400 credits.
**Why PostgreSQL:** Free forever, runs in Docker, needed for application layer (alerts, tracking, evaluations).
**Tradeoff:** Snowflake trial expires. Accepted because project builds in 6 weeks, well within trial.

### D4: Why Mistral-7B and not GPT-4 or Llama?
**Decision:** Fine-tune Mistral-7B-Instruct with QLoRA.
**Why Mistral:** Fits on Colab free T4 GPU with 4-bit quantization. Open-source. Strong instruction-following. Active community.
**Why not GPT-4:** Can't fine-tune GPT-4. Also costs money per call.
**Why not Llama-3:** Larger model, harder to fit on free Colab. Mistral-7B is the sweet spot for quality vs. compute.
**Tradeoff:** 7B model won't match GPT-4 quality. Accepted because the point is demonstrating fine-tuning skill, not building the best summarizer.

### D5: Why LLM-as-Judge and not human evaluation?
**Decision:** Automated evaluation using LLM-as-Judge + RAGAS metrics.
**Why:** Scalable (can evaluate 200 products automatically), reproducible, industry-standard approach (Capital One, Deloitte JDs mention evaluation frameworks).
**Tradeoff:** LLM judges have biases (prefer verbose answers, favor their own style). Mitigated by using structured rubric with specific criteria (accuracy, completeness, actionability, conciseness).
**Interview talking point:** "I chose automated evaluation for scalability, but acknowledge LLM judge limitations — I'd complement with human evaluation in a production setting."

### D6: Why create labels manually instead of using existing labeled datasets?
**Decision:** LLM-assisted labeling of 3000 reviews + manual verification of 500.
**Why:** No existing dataset has root cause labels (defect, shipping, description, size, price) for Amazon reviews. This is the reality of real-world DS work.
**Tradeoff:** Labels may have noise from LLM. Mitigated by manual verification of 500 samples.
**Interview talking point:** "I used a semi-supervised labeling approach — LLM for initial annotation, human verification for quality control. This mirrors how production ML teams handle label scarcity."

### D7: Why A/B test base vs. fine-tuned (not fine-tuned vs. GPT-4)?
**Decision:** Compare base Mistral-7B vs. QLoRA fine-tuned Mistral-7B.
**Why:** This proves fine-tuning adds value. Comparing against GPT-4 is unfair (different model size) and doesn't demonstrate YOUR contribution.
**What we measure:** 200 test products, both models generate summaries, LLM-as-Judge scores both, two-sample t-test for statistical significance.
**Target:** p < 0.05 with fine-tuned model scoring higher.

### D8: Why LangGraph and not CrewAI or AutoGen?
**Decision:** LangGraph for multi-agent orchestration.
**Why:** Capital One's Applied GenAI DS role explicitly names LangGraph. It's the most production-ready agent framework. State management is built-in. Already using it in PE OrgAIR project.
**Tradeoff:** Steeper learning curve than CrewAI. Accepted because you already have LangGraph experience.

---

## Technical Decisions (fill during build)

### T1: [Step 1.2] — ChromaDB Cloud instead of Docker [REVISED]
~~**Decision:** Use ChromaDB Cloud (free tier) instead of running ChromaDB in Docker.~~
**Revised Decision:** ChromaDB back in Docker alongside PostgreSQL.
**Why revised:** ChromaDB Cloud free credits ($4 remaining) insufficient for 500K embeddings at 384 dimensions. Docker ChromaDB is lightweight (~200MB RAM) and PostgreSQL container was already running fine, so adding ChromaDB doesn't meaningfully impact performance.
**Tradeoff:** Slightly heavier Docker setup, but eliminates cloud dependency and cost risk.

### T2: [Step 2.1] — Stream directly into PostgreSQL, skip local parquet files
**Decision:** Stream data from HuggingFace directly into PostgreSQL using streaming mode. No local parquet files stored.
**Alternatives:** Download full dataset to local parquet (original plan), download subset to parquet then load.
**Why:** Dev machine has ~12 GB free disk. Full Electronics dataset is 22.6 GB. Even 500K reviews as parquet (~300 MB) failed due to low disk. Streaming avoids all local storage — data goes HuggingFace → memory (10K batch) → PostgreSQL.
**Tradeoff:** No local raw data backup. Acceptable because HuggingFace is the source of truth (free, always available) and Snowflake will serve as the warehouse backup. Can re-stream anytime with `poetry run python src/data/download.py`.

### T3: [Step 2.1] — 500K review subset for development
**Decision:** Use 500K reviews instead of full 20M for development and model training.
**Alternatives:** Full 20M dataset.
**Why:** Disk space constraint. Model training only needs labeled subsets (3K for classifier). 500K is statistically representative for EDA, features, and all ML tasks.
**Tradeoff:** Smaller numbers in EDA/dashboard. Interview defense: "Pipeline scales to 20M — I validated on 500K for iteration speed."

### T4: [Step 2.4] — Regex NER instead of spaCy EntityRuler for bulk processing
**Decision:** Use pre-compiled regex for bulk NER extraction (500K reviews), keep spaCy for demos/tests.
**Alternatives:** spaCy EntityRuler (original), LLM-based extraction (Groq/Claude), FlashText.
**Why:** spaCy EntityRuler took 38 min per 50K reviews (~6 hours total). Regex does 500K in 2 minutes. LLM-based would cost money and take days. Same pattern matching results, 100x faster.
**Tradeoff:** No time_refs extraction in fast mode (regex skips date parsing). Acceptable — time_refs aren't used by any downstream model.

### T5: [Step 3.1] — Groq (free) for root cause labeling instead of Claude API
**Decision:** Use Groq's Llama 3.1-8B (free tier, 30 req/min) to label 3K negative reviews.
**Alternatives:** Claude API ($3+/1M tokens), manual labeling, GPT-4.
**Why:** Only 3K reviews need labeling — straightforward 5-category classification. Groq is free and fast enough. Save paid APIs for Phase 5 (LLM-as-Judge) where quality matters more.
**Tradeoff:** ~100 min wall time due to rate limiting. Lower quality than Claude/GPT-4 but sufficient for training labels — the classifier learns the pattern, not the exact label.

### T6: [Step 3.1] — 3000 labeled samples for classifier training
**Decision:** Label 3000 negative reviews (not more, not fewer).
**Alternatives:** 1500 (faster), 5000+ (more data), manual labeling.
**Why:** Research shows diminishing returns beyond 3K for 5-category classification with DistilBERT. After 70/15/15 split: 2100 train, 450 val, 450 test — sufficient for Macro-F1 > 0.70 target.
**Tradeoff:** More labels might squeeze out extra F1 points, but 2x the Groq time for marginal gain.

### T7: [Step 3.2] — Class weights + threshold tuning for classifier
**Decision:** Use inverse-frequency class weights in BCEWithLogitsLoss and tune decision threshold on val set (0.3-0.5 range).
**Alternatives:** Oversample rare classes (SMOTE), collect more labels, use focal loss.
**Why:** First run hit F1=0.67 (target 0.70). Shipping (171/3000 = 5.7%) and size (243/3000 = 8.1%) are rare — model under-predicts them. Class weights penalize missing rare categories more. Threshold tuning finds optimal cutoff per the val set instead of fixed 0.5.
**Tradeoff:** Higher recall for rare classes may slightly reduce precision for defect (dominant class). Acceptable — macro F1 weights all classes equally so improving rare classes helps more than a small defect precision drop.

### T8: [Step 3.2-3.4] — Local MLflow file tracking instead of server
**Decision:** Use `mlflow.set_tracking_uri("file:./mlruns")` for local file-based tracking.
**Alternatives:** Run MLflow server in Docker, use MLflow sqlite backend.
**Why:** MLflow server wasn't running and adding another Docker container is unnecessary for development. File-based tracking works identically — can still view with `mlflow ui`.
**Tradeoff:** No concurrent access or remote viewing. Fine for single-developer project.

<!-- Copy this template for each technical decision during build:

### T9: [Step 3.2] — Extra shipping labels to fix class imbalance
**Decision:** Targeted 500 additional shipping-keyword reviews from PostgreSQL and labeled them with Groq. Merged into main training set (3000→3500 labels, shipping 171→344).
**Alternatives:** Oversample existing shipping labels (SMOTE), synthetic data generation, accept lower F1.
**Why:** Shipping had only 171 labels (5.7%) — too few for the model to learn. Instead of synthetic augmentation, we found real reviews mentioning shipping keywords and labeled them. Real data > synthetic data.
**Tradeoff:** Used another Groq API key's daily limit. Worth it — classifier val F1 jumped from 0.69 to 0.7255.

### T10: [Step 4.1] — 50K embeddings, negative-first priority
**Decision:** Embed 50K reviews (rating <= 3) into ChromaDB using all-MiniLM-L6-v2 (384-dim vectors).
**Alternatives:** Embed all 500K reviews, use a larger model (e5-large).
**Why:** Semantic search is primarily used for finding complaint patterns — negative/mixed reviews matter most. 50K keeps ChromaDB fast and fits comfortably in Docker. MiniLM is 5x faster than larger models with 95% quality.
**Tradeoff:** Positive reviews (4-5 stars) not searchable. Acceptable — the use case is complaint analysis, not positive review search.

### T11: [Step 5.1] — Claude Sonnet for training data instead of Groq Llama 8B
**Decision:** Regenerated 400 summary training pairs using Claude Sonnet instead of Groq Llama 3.1-8B.
**Alternatives:** Groq Llama 3.1-70B (free), keep Llama 8B data, manual writing.
**Why:** First fine-tuning attempt with Llama 8B data scored 3.12/5 (worse than base 3.37). Training data quality was the bottleneck. Claude Sonnet produces significantly better structured summaries. Cost: ~$4.20.
**Tradeoff:** Used most of the $10 budget. Worth it — summary quality visibly improved (specific numbers, accurate sentiment, clean structure).

### T12: [Step 5.2] — Lean QLoRA config for T4 memory
**Decision:** Reduced LoRA rank from 16→8, target modules from 4→2 (q,v_proj only), batch size 2→1, added gradient checkpointing, 3→2 epochs.
**Alternatives:** Use A100 ($10 Colab Pro), keep original config and risk OOM.
**Why:** Original config OOM'd at epoch 8 after 5 hours on T4 (16GB). Lean config fits comfortably. Quality impact minimal — r=8 with q,v_proj is the standard QLoRA config from the original paper.
**Tradeoff:** Slightly less model capacity. Negligible for structured summarization task.

### T13: [Step 5.4] — A/B test showed no significant improvement
**Decision:** Document as valid negative result rather than gaming the metrics.
**Alternatives:** Use Claude as judge (might show different results), increase training data to 2000+, try different hyperparameters.
**Why:** Fine-tuned (3.90/5) vs base (3.94/5) — no statistically significant difference (p=0.72). The pipeline is correct. The result is honest. With 400 training pairs and 2 epochs, Mistral-7B couldn't significantly outperform the base. In production: 2000+ human-verified pairs + stronger base model.
**Tradeoff:** Two targets failed (A/B p-value, judge avg >4.0). Interview defense: "Not every experiment succeeds. I built the full evaluation pipeline to detect this. The 3 PyTorch models all passed."

### T14: [Step 7.3] — Evidently embedded in dashboard, not separate service
**Decision:** Run Evidently drift check as a script, embed HTML report directly in Streamlit dashboard.
**Alternatives:** Run Evidently as a separate monitoring service, use Grafana for drift visualization.
**Why:** Separate monitoring service adds infrastructure complexity for a portfolio project. Embedding the report in the dashboard keeps everything in one place — a PM can see model metrics AND drift in the same UI.
**Tradeoff:** No real-time drift alerting. Acceptable — drift checks run on-demand or scheduled, not continuously.

### T15: [Step 8.1] — Rule-based classifier demo instead of live model inference
**Decision:** Dashboard classifier demo uses keyword matching against NER patterns, not the actual DistilBERT model.
**Alternatives:** Load DistilBERT model in Streamlit, call FastAPI /classify endpoint.
**Why:** Loading DistilBERT in Streamlit doubles memory usage and adds 10s startup time. The keyword-based demo shows the same categories and is instant. The real model runs in the training/evaluation pipeline.
**Tradeoff:** Demo scores are approximate (keyword-based), not exact model predictions. Made this explicit in the UI.

-->

---

## Interview-Ready Tradeoff Answers

**"Why didn't you use a larger model?"**
→ "I chose Mistral-7B because it fits on free GPU (Colab T4) with 4-bit quantization. In production, I'd scale to a 70B model, but the fine-tuning methodology and evaluation framework I built transfer directly."

**"Why not use a pre-built sentiment analysis API?"**
→ "I trained my own PyTorch classifier because the goal was demonstrating model training skills, not just calling an API. Plus, pre-built APIs don't do multi-label root cause classification — they only do positive/negative."

**"How would you scale this to handle real-time reviews?"**
→ "Current design is batch. For real-time, I'd add Kafka for streaming ingestion, a model serving layer (TorchServe or Triton), and Redis caching for frequently queried products. The core models and agents wouldn't change."

**"Why not use a feature store like Feast?"**
→ "Feature computation here is straightforward (aggregations over time windows). Feast adds complexity without proportional value for this use case. In a production setting with multiple teams consuming features, I'd absolutely add it."

**"Your anomaly detector is simple. Why not use a more sophisticated approach?"**
→ "I used an autoencoder because it's interpretable and sufficient for detecting distribution shifts in review patterns. I considered Isolation Forest and Prophet, but the autoencoder gives me a reconstruction error that's more explainable to product managers. In production, I'd ensemble multiple approaches."

**"Why not use an LLM to extract entities from reviews?"**
→ "For 500K reviews, LLM extraction would take days and cost money. Pre-compiled regex does it in 2 minutes with identical results for our pattern set. LLMs shine when nuance matters — that's why I use Groq for the 3K labeling task where the categories are subjective. Right tool for the right job."

**"How did you create your training labels?"**
→ "LLM-assisted labeling — I used Groq's Llama 3.1 to classify 3K negative reviews into 5 root cause categories. This is a realistic approach: no off-the-shelf dataset has these labels, and manual labeling 3K reviews would take days. The LLM gives us consistent labels at scale, and the trained classifier then generalizes to all 500K reviews."

**"Why DistilBERT and not a larger model for classification?"**
→ "DistilBERT is 6x faster than BERT with 97% of its performance on classification tasks. For a 5-label classifier on short review texts, the bottleneck is training data quality, not model capacity. I hit Macro-F1 > 0.70 without needing a heavier model."

**"Why three separate models instead of one multi-task model?"**
→ "Each model solves a fundamentally different problem: anomaly detection is unsupervised, classification is supervised multi-label, helpfulness is regression. Different loss functions, different data, different evaluation metrics. A multi-task model adds coupling complexity for no real gain. In production, separate models are also easier to update independently."

**"Your fine-tuned model didn't beat the base. Why?"**
→ "With 400 training pairs and 2 epochs on a 7B model, the fine-tuning was too shallow to create a significant quality gap. The model learned the format but not substantially better content. In production, I'd use 2000+ human-verified training pairs, train longer, and use a stronger judge model. The important thing is I built the full pipeline — QLoRA fine-tuning, LLM-as-Judge evaluation, and statistical A/B testing — to detect this."

**"Why use LangGraph instead of simple function chaining?"**
→ "LangGraph gives us state management, conditional routing (skip rewriter if no mismatches), error handling per node, and a visual graph representation. Simple chaining works for a linear pipeline, but our pipeline has conditional logic — the auditor decides whether the rewriter runs. LangGraph makes this explicit and extensible."

**"Why 4 agents instead of one big prompt?"**
→ "Separation of concerns. Each agent has a focused job with clear inputs/outputs. The Analyzer doesn't need to know about listing rewriting. The Supervisor doesn't need to run NER. This makes each agent testable independently, debuggable, and replaceable. If we upgrade the rewriter to GPT-4, we change one file."