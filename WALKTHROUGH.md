# Project Walkthrough — Interview Prep

Step-by-step explanation of what was built, why, and how. Read this to walk an interviewer through the project from start to finish.

---

## Architecture Overview — The Big Picture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        HOW DATA FLOWS                               │
 │                                                                     │
 │  HuggingFace ──stream──► PostgreSQL ◄── Snowflake (warehouse)       │
 │                              │                                      │
 │          ┌───────────────────┼───────────────────┐                  │
 │          ▼                   ▼                   ▼                  │
 │     spaCy NER         Feature Pipeline     Sentence Transformers   │
 │     (extract           (rolling stats        (384-dim vectors)     │
 │      components         per product)              │                │
 │      & issues)              │                     ▼                │
 │          │                  │               ChromaDB               │
 │          ▼                  ▼               (semantic search)      │
 │     Root Cause         Anomaly Detector                            │
 │     Classifier         (quality alerts)     Helpfulness            │
 │     (why angry?)            │               Predictor              │
 │          │                  │               (useful review?)       │
 │          ▼                  ▼                                      │
 │     ┌───────────────────────────────────────┐                      │
 │     │         LangGraph Agent Pipeline       │                      │
 │     │                                        │                      │
 │     │   Analyzer ──► Auditor ──► Rewriter    │                      │
 │     │   "What's      "Listing    "Fix the    │                      │
 │     │    wrong?"      matches?"   listing"   │                      │
 │     │         │                              │                      │
 │     │         ▼                              │                      │
 │     │     Supervisor (human approval)        │                      │
 │     └───────────────────────────────────────┘                      │
 │                         │                                           │
 │                         ▼                                           │
 │                    FastAPI Backend                                   │
 │                         │                                           │
 │                         ▼                                           │
 │                  Streamlit Dashboard                                 │
 │           (Alerts | Product Deep Dive |                             │
 │            Classifier Demo | Model Perf)                            │
 └─────────────────────────────────────────────────────────────────────┘
```

### How to explain the flow in 2 minutes:

> "Reviews come in from Amazon. My pipeline first extracts entities — what product component is mentioned and what's the issue. Then I compute daily features per product: sentiment trends, review velocity, complaint ratios.
>
> Three PyTorch models run on this data:
> 1. **Anomaly detector** catches products where something just went wrong — sudden spike in negative reviews
> 2. **Root cause classifier** tells you WHY — is it a defect? shipping damage? misleading description?
> 3. **Helpfulness predictor** scores which reviews are most actionable
>
> For products with issues, a 4-agent LangGraph pipeline kicks in:
> - **Analyzer** pulls the review data and complaint profile
> - **Auditor** compares the product listing against actual complaints — 'listing says 8-hour battery but reviews say 4 hours'
> - **Rewriter** uses fine-tuned Mistral-7B to generate an improved listing
> - **Supervisor** holds it for human approval
>
> Everything is served through FastAPI and displayed on a Streamlit dashboard where a product manager can see alerts, drill into any product, and approve listing changes."

### Why this architecture (interview talking points):

- **Two databases:** Snowflake for analytics at scale (data team queries), PostgreSQL for fast app reads/writes (API + dashboard). This is how production systems work.
- **Three separate models, not one:** Each solves a different problem with different data and evaluation metrics. Shows you can design ML systems, not just call `.fit()`.
- **LangGraph agents, not a single LLM call:** The agent pipeline has state management, conditional routing (skip rewrite if no anomaly), and human-in-the-loop. This is production-grade agentic AI.
- **Fine-tuned LLM with evaluation:** Not just "I used GPT-4." I fine-tuned Mistral-7B with QLoRA, evaluated with LLM-as-Judge, and proved improvement with statistical A/B testing (p < 0.05).

---

## Phase 1: Project Setup

**What we did:**
- Initialized a Python 3.11 project with Poetry for dependency management
- Set up pre-commit hooks (black, isort, flake8) so every commit is auto-formatted and linted
- Created a Dockerfile and docker-compose.yml with PostgreSQL 16 and ChromaDB
- Initialized DVC (Data Version Control) with a pipeline: download → extract features → train models

**Why it matters:**
- Production teams use these exact tools. This isn't a Jupyter notebook thrown on GitHub — it's a properly structured ML project.
- DVC tracks the ML pipeline so you can reproduce any stage with `dvc repro`.

**Flow:** Poetry manages Python deps → Docker runs databases → DVC tracks the ML pipeline → pre-commit ensures code quality on every commit.

---

## Phase 2: Data Pipeline

**Flow:** HuggingFace → (stream) → PostgreSQL → spaCy NER → Feature Pipeline → Ready for models

### Step 2.1 — Data Acquisition
**What we did:**
- Streamed Amazon Reviews 2023 dataset from HuggingFace directly into PostgreSQL (no local files — disk space constraint)
- Used the Electronics category — 500K reviews for development, pipeline supports full 20M
- Also streamed product metadata (title, description, price, features, ratings)
- Batch inserts of 10K rows at a time for efficiency

**Why this dataset:**
- Free, high-quality, real-world data (571M reviews total)
- Clear business impact: e-commerce companies lose $50B+/year on returns
- No existing dataset has root cause labels — we had to create our own (realistic ML work)

**How to explain:** "I chose a large-scale, real-world dataset to simulate production conditions. The pipeline streams data directly into PostgreSQL to avoid local storage overhead — this mirrors how production pipelines ingest data from external sources."

### Step 2.2 — Snowflake Data Warehouse
**What we did:**
- Created database PRODUCT_INTELLIGENCE with schema RAW in Snowflake
- Bulk loaded reviews and product metadata
- Tables: REVIEWS (rating, title, text, helpful_votes, timestamp, asin) and PRODUCTS (asin, title, description, price, features)

**Why Snowflake:**
- Industry-standard cloud data warehouse (used at most Fortune 500 companies)
- Handles the full 20M review dataset without breaking a sweat
- Demonstrates you can work with enterprise data infrastructure, not just pandas on a laptop

**How to explain:** "Snowflake is the data warehouse layer — it stores the full raw dataset for analytical queries. The application layer (PostgreSQL) holds a working subset for fast reads. This mirrors real production architecture where analytics queries hit the warehouse and the app hits a transactional DB."

**Flow:** HuggingFace → Snowflake (full dataset for analytics) AND HuggingFace → PostgreSQL (500K subset for the app)

### Step 2.3 — PostgreSQL Application Database
**What we did:**
- Set up PostgreSQL in Docker with SQLAlchemy ORM
- Created tables: reviews, products, alerts, application_tracker
- Streamed 500K reviews directly from HuggingFace
- Added compound indexes on (asin, timestamp) and (asin, rating) for fast queries

**Why this design:**
- `reviews` + `products` = source data for models and agents
- `alerts` = output of anomaly detector (quality issues found)
- `application_tracker` = tracks which products have been analyzed
- Indexes make product-level queries fast (the dashboard queries by product constantly)

**Flow:** All downstream processing reads from PostgreSQL → NER, features, models, agents

### Step 2.4 — spaCy NER (Named Entity Recognition)
**What we did:**
- Built a custom NER pipeline using spaCy with EntityRuler
- Extracts three entity types from review text:
  - **Product components:** battery, screen, bluetooth, charger, USB, speaker, etc. (40+ patterns)
  - **Issue types:** broken, defective, disconnecting, overheating, slow, etc. (38+ patterns)
  - **Time references:** dates and time mentions (built-in spaCy NER)
- Used token-level LOWER matching for case-insensitive extraction
- Batch processing with spaCy's `pipe()` for speed

**Why custom NER, not just keywords:**
- spaCy's EntityRuler handles multi-word patterns ("hard drive", "stopped working")
- Integrates with spaCy's built-in NER for time references
- Extensible — can add ML-based entity recognition later
- Production-grade: processes thousands of reviews per second with `pipe()`

**How to explain:** "I built a custom entity extraction pipeline because no off-the-shelf NER model knows that 'battery' is a product component in the context of electronics reviews. The EntityRuler gives us pattern-based extraction with spaCy's speed."

**Flow:** Review text → spaCy NER → {components: ["battery", "screen"], issues: ["broken", "overheating"]} → used by feature pipeline AND agents

### Step 2.5 — Feature Engineering
**What we did:**
- For each product, computed daily rolling features:
  - `daily_sentiment_avg` — 7-day rolling average of ratings
  - `review_velocity` — reviews per day (7-day rolling)
  - `negative_ratio` — % of 1-2 star reviews in last 7 days
  - `complaint_keywords` — count of negative keywords per day
  - `ner_entities` — aggregated NER output (which components/issues mentioned most)
- Stored in PostgreSQL `product_features` table

**Why these features:**
- These are the inputs to the anomaly detection model
- A sudden spike in `negative_ratio` or `review_velocity` signals a product quality issue
- Rolling windows smooth out noise and capture trends, not one-off bad reviews

**How to explain:** "These features answer the question: 'Is something going wrong with this product right now?' A spike in negative ratio combined with increased review velocity is a strong signal for a quality alert."

**Flow:** PostgreSQL reviews → rolling aggregation per product per day → `product_features` table → anomaly detector reads from here

### Step 2.6 — Exploratory Data Analysis
**What we did:**
- Built a Jupyter notebook with 8 analysis sections:
  1. Data overview (total reviews, products)
  2. Rating distribution
  3. Reviews over time (monthly trend)
  4. Top 20 most reviewed products
  5. Word cloud of negative reviews
  6. Sample NER extractions on real reviews
  7. Review length distribution
  8. Helpful votes distribution

**Why it matters:**
- Shows you understand the data before building models
- The word cloud and NER samples validate that the text processing pipeline is working
- Presentation-ready plots (matplotlib + seaborn) — not raw pandas output

---

## Phase 3: ML Models

**Flow:** PostgreSQL features → 3 PyTorch models → MLflow (tracking) → model checkpoints

### Step 3.1 — Root Cause Labels
**What we did:** [TBD]

**Flow:** 3000 negative reviews → LLM labels them → manual verification of 500 → `labeled_reviews.csv` → classifier training data

### Step 3.2 — Root Cause Classifier
**What we did:** [TBD]

**Flow:** Labeled reviews → DistilBERT + BCEWithLogitsLoss → multi-label output: [defect, shipping, description, size, price] → MLflow logs metrics

### Step 3.3 — Anomaly Detector
**What we did:** [TBD]

**Flow:** `product_features` table → autoencoder trains on "normal" products → reconstruction error = anomaly score → threshold at 95th percentile → high score = quality alert

### Step 3.4 — Helpfulness Predictor
**What we did:** [TBD]

**Flow:** Review features (length, rating, NER count, etc.) → neural network → predicts helpful_votes → surfaces most actionable reviews

### Step 3.5 — MLflow Experiments
**What we did:** [TBD]

**Flow:** All 3 models log to MLflow → compare experiments → register best models → tag as "staging"

---

## Phase 4: Embeddings & Vector Search

**Flow:** Review text → sentence-transformers → 384-dim vectors → ChromaDB → semantic search via FastAPI

### Step 4.1 — Generate Embeddings
**What we did:** [TBD]

### Step 4.2 — Semantic Search
**What we did:** [TBD]

**Flow:** User query "bluetooth issues" → encode → ChromaDB similarity search → return top-K matching reviews

---

## Phase 5: LLM Fine-tuning & Evaluation

**Flow:** Training pairs → QLoRA fine-tune Mistral-7B → LLM-as-Judge evaluation → A/B test (base vs fine-tuned) → statistical significance

### Step 5.1 — Training Data
**What we did:** [TBD]

**Flow:** 200 products with 50+ reviews → group by complaint category → LLM generates structured summary → 1500 (input, summary) pairs

### Step 5.2 — QLoRA Fine-tuning
**What we did:** [TBD]

**Flow:** Base Mistral-7B → 4-bit quantization → LoRA adapters on q_proj, v_proj → train 3 epochs on Colab T4 → save adapter weights

### Step 5.3 — LLM-as-Judge
**What we did:** [TBD]

**Flow:** For each product: reviews + generated summary → evaluator LLM scores on accuracy, completeness, actionability, conciseness (1-5 each) → store in PostgreSQL

### Step 5.4 — A/B Testing
**What we did:** [TBD]

**Flow:** 200 test products × 2 models → LLM judge scores both → two-sample t-test per metric → p < 0.05 = statistically significant improvement

---

## Phase 6: Agentic AI (LangGraph)

**Flow:** Product ID → Analyzer → Auditor → Rewriter → Supervisor (human approval) → done

### Step 6.1-6.4 — Agent Pipeline
**What we did:** [TBD]

**How to explain the agent flow:**
> "When a product is flagged, four agents run in sequence:
> 1. **Analyzer** pulls all reviews, runs them through the classifier, and builds a complaint profile — '45% defect, 30% description mismatch'
> 2. **Auditor** compares the product listing against the complaint profile — finds mismatches like 'listing says 8-hour battery, but 60% of reviews say 4-5 hours'
> 3. **Rewriter** uses fine-tuned Mistral to generate an improved listing with honest disclaimers
> 4. **Supervisor** holds it for human review — no changes go live without approval
>
> If the anomaly score is below threshold, the pipeline short-circuits after the Analyzer — no rewrite needed."

---

## Phase 7: API & CI/CD

**Flow:** All functionality → FastAPI endpoints → GitHub Actions validates on every push → Evidently monitors for drift

*(To be filled as we build)*

---

## Phase 8: Dashboard & Polish

**Flow:** FastAPI backend → Streamlit reads from API → 4 pages (Alerts, Deep Dive, Classifier Demo, Model Performance)

*(To be filled as we build)*
