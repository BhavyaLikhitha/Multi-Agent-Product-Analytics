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

### T1: [Step 1.2] — ChromaDB Cloud instead of Docker
**Decision:** Use ChromaDB Cloud (free tier) instead of running ChromaDB in Docker.
**Alternatives:** ChromaDB in docker-compose (original plan), ChromaDB in-memory (no persistence).
**Why:** Docker with multiple services slows down the dev machine. ChromaDB Cloud free tier provides persistent vector storage without local resource usage. PostgreSQL remains in Docker since it's lightweight and needed locally.
**Tradeoff:** Adds external dependency (network latency, cloud account required). Accepted because it keeps the dev environment fast and ChromaDB Cloud free tier is sufficient for this project's scale.

<!-- Copy this template for each technical decision during build:

### TN: [Step X.X] — [Decision title]
**Decision:** 
**Alternatives:** 
**Why:** 
**Tradeoff:** 

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