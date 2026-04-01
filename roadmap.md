# Build Roadmap — Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring
# Step-by-step instructions for building with Claude Code

---

## HOW TO USE THIS FILE
- Follow steps IN ORDER. Don't skip ahead.
- Each step = one Claude Code session/prompt.
- Copy the prompt, paste into Claude Code, let it build.
- Test each step before moving to next.
- Check off completed steps.

---

## PHASE 1: PROJECT SETUP (Day 1)

### Step 1.1 — Initialize repo
Prompt: "Create a Python project called `product-review-intelligence`. Set up:
- Poetry for dependency management
- Python 3.11
- Folder structure: src/, tests/, notebooks/, data/, models/, configs/
- .gitignore for Python, data files, model files, .env
- Empty README.md
- .env.example with placeholder env vars
- Makefile with commands: install, test, lint, format
- Pre-commit hooks for black, isort, flake8
Initialize git repo."

### Step 1.2 — Set up Docker foundation
Prompt: "Add Docker setup:
- Dockerfile for the Python app
- docker-compose.yml with services: app, postgres, chromadb
- PostgreSQL 16 with volume mount
- ChromaDB with volume mount
- Shared network between all services
- Health checks for each service
Test with: docker-compose up -d"

### Step 1.3 — Set up DVC
Prompt: "Initialize DVC in the project:
- dvc init
- Create data/raw/ and data/processed/ directories
- Add .dvc files to git
- Create a DVC pipeline file (dvc.yaml) with placeholder stages:
  stage 1: download_data
  stage 2: preprocess
  stage 3: extract_features
  stage 4: train_models
Configure local storage for now (no remote)."

---


## PHASE 2: DATA PIPELINE (Days 2-4)

### Step 2.1 — Download Amazon Reviews dataset
Prompt: "Write a Python script src/data/download.py that:
- Uses HuggingFace datasets library to download McAuley-Lab/Amazon-Reviews-2023
- Downloads only the 'Electronics' category (raw_review_Electronics)
- Also downloads product metadata (raw_meta_Electronics)
- Saves as parquet files in data/raw/
- Logs download progress
- Add this as the 'download_data' stage in dvc.yaml
Test: run the script and verify parquet files exist."

### Step 2.2 — Load into Snowflake
Prompt: "Write src/data/load_snowflake.py that:
- Connects to Snowflake using snowflake-connector-python
- Creates database PRODUCT_INTELLIGENCE
- Creates schema RAW
- Creates tables: REVIEWS (rating, title, text, helpful_votes, timestamp, asin, user_id) and PRODUCTS (asin, title, description, price, category, features, average_rating)
- Bulk loads parquet files into these tables using COPY INTO or pandas
- Reads Snowflake credentials from .env
- Prints row counts after loading
Use staging with PUT if needed."

### Step 2.3 — Load into PostgreSQL (application DB)
Prompt: "Write src/data/load_postgres.py that:
- Connects to PostgreSQL (from docker-compose)
- Uses SQLAlchemy as ORM
- Creates tables: reviews, products, alerts, application_tracker
- Loads a SUBSET of reviews (latest 1M) for the application layer
- The alerts table has: id, product_id, alert_type, severity, detected_at, details
- The application_tracker table has: id, product_id, status, last_analyzed, summary
- Indexes on product_id, timestamp, rating for fast queries
Test: run the script, verify with a SQL query."

### Step 2.4 — spaCy NER pipeline for entity extraction
Prompt: "Write src/features/ner_extractor.py that:
- Uses spaCy (en_core_web_sm or en_core_web_trf)
- Takes a review text as input
- Extracts: product_component (e.g. 'battery', 'screen', 'bluetooth'), issue_type (e.g. 'broke', 'disconnecting', 'slow'), quantity/time references
- For custom product entities, use spaCy's EntityRuler with patterns for common e-commerce terms (battery, screen, charger, cable, bluetooth, wifi, speaker, etc.)
- Returns structured dict: {components: [...], issues: [...], time_refs: [...]}
- Process in batches using spaCy's pipe() for speed
- Write a test in tests/test_ner.py with 10 sample reviews
Test: run on 100 sample reviews, verify extraction quality."

### Step 2.5 — Feature engineering pipeline
Prompt: "Write src/features/feature_pipeline.py that:
- Reads reviews from PostgreSQL
- For each product, computes:
  - daily_sentiment_avg (rolling 7-day average of ratings)
  - review_velocity (reviews per day, 7-day rolling)
  - negative_ratio (% of 1-2 star reviews in last 7 days)
  - complaint_keywords (count of negative keywords per day)
  - ner_entities (aggregated NER output per product per day)
- Stores computed features back in PostgreSQL in a 'product_features' table
- Tracks feature computation in DVC
This is the input for the anomaly detection model."

### Step 2.6 — EDA notebook
Prompt: "Create notebooks/01_eda.ipynb that:
- Connects to PostgreSQL
- Shows: total reviews, total products, rating distribution, reviews over time
- Top 20 most reviewed products
- Word cloud of negative reviews (1-2 stars)
- Sample NER extractions on 20 reviews
- Distribution of review lengths
- Helpful votes distribution
- This notebook should tell a clear story about the data quality and what patterns exist
Use matplotlib and seaborn. Clean, presentation-ready plots."

---

## PHASE 3: ML MODELS (Days 5-10)

### Step 3.1 — Create root cause labels (LLM-assisted)
Prompt: "Write src/data/create_labels.py that:
- Samples 3000 negative reviews (1-2 stars) from PostgreSQL
- For each review, calls an LLM (use OpenAI API or Anthropic API) with this prompt:
  'Classify this product review into one or more categories: product_defect, shipping_damage, misleading_description, size_fit_issue, price_complaint, other. Return only the categories as a comma-separated list.'
- Saves results as data/processed/labeled_reviews.csv with columns: review_id, text, rating, labels
- Includes a manual verification script that shows 20 random labeled reviews for human spot-check
- Track this labeled dataset with DVC"

### Step 3.2 — Train root cause classifier (PyTorch)
Prompt: "Write src/models/root_cause_classifier.py that:
- Loads labeled_reviews.csv
- Uses HuggingFace transformers: DistilBERT for multi-label classification
- Multi-label setup: BCEWithLogitsLoss, sigmoid outputs
- Train/val/test split: 70/15/15
- Custom PyTorch training loop (NOT Trainer API — we want to show we can write training loops)
- Learning rate scheduler: linear warmup + cosine decay
- Logs to MLflow: hyperparams, per-epoch loss, per-label F1, macro-F1, confusion matrix
- Saves best model checkpoint to models/root_cause/
- Evaluation script that prints per-label precision, recall, F1
- Target: macro-F1 > 0.7
Write tests/test_classifier.py with basic model inference test."

### Step 3.3 — Train anomaly detector (PyTorch)
Prompt: "Write src/models/anomaly_detector.py that:
- Loads product_features from PostgreSQL (daily sentiment, velocity, negative ratio per product)
- Trains a PyTorch autoencoder:
  - Encoder: 3 layers (input_dim → 64 → 32 → 16)
  - Decoder: 3 layers (16 → 32 → 64 → input_dim)
  - Loss: MSE reconstruction loss
- Train on 'normal' periods (products with stable ratings)
- Anomaly score = reconstruction error
- Threshold: set at 95th percentile of training reconstruction errors
- Logs to MLflow: loss curves, threshold, sample anomalies detected
- Saves model to models/anomaly/
- Evaluation: run on known anomalous products (manually identified), compute precision/recall
Write tests/test_anomaly.py."

### Step 3.4 — Train helpfulness predictor (PyTorch)
Prompt: "Write src/models/helpfulness_predictor.py that:
- Loads reviews with helpful_votes from PostgreSQL
- Features: review length, rating, number of sentences, has_specific_numbers (bool), has_comparison (bool), sentiment_score, NER_entity_count
- Target: helpful_votes (regression) or helpful_binary (classification: >5 votes = helpful)
- PyTorch neural network: 3 hidden layers with ReLU, dropout 0.3, batch norm
- Custom training loop with MLflow logging
- Metrics: MAE, RMSE for regression or F1 for classification
- Saves to models/helpfulness/
Write tests/test_helpfulness.py."

### Step 3.5 — MLflow experiment dashboard
Prompt: "Set up MLflow properly:
- Add mlflow service to docker-compose.yml (mlflow server with PostgreSQL backend)
- Update all 3 model training scripts to log to this MLflow server
- Create a script src/mlops/compare_models.py that:
  - Queries MLflow for all runs
  - Prints comparison table of all experiments
  - Registers best model of each type in MLflow Model Registry
  - Tags them as 'staging'
- Verify: open MLflow UI at localhost:5000 and see all experiments"

---

## PHASE 4: EMBEDDINGS & VECTOR SEARCH (Days 11-12)

### Step 4.1 — Generate review embeddings
Prompt: "Write src/features/generate_embeddings.py that:
- Uses sentence-transformers (all-MiniLM-L6-v2) to encode review text
- Processes reviews in batches of 512
- Generates 384-dim embeddings for each review
- Stores in ChromaDB with metadata: product_id, rating, timestamp
- Progress bar with tqdm
- Process the full 1M review subset
- Track with DVC"

### Step 4.2 — Semantic search endpoint
Prompt: "Write src/api/search.py that:
- Takes a natural language query: 'bluetooth connectivity issues'
- Searches ChromaDB for top-K similar reviews
- Returns: review text, rating, product_id, similarity score
- Also supports: 'find all reviews for product X mentioning Y'
- Write a simple test that queries 5 different complaint types"

---

## PHASE 5: LLM FINE-TUNING & EVALUATION (Days 13-17)

### Step 5.1 — Curate summary training data
Prompt: "Write src/data/create_summary_pairs.py that:
- For each of 200 products with 50+ reviews:
  - Pulls all reviews from PostgreSQL
  - Groups into complaint categories using the root cause classifier
  - Calls base Mistral (or GPT-4-mini) to generate a structured summary:
    'Top complaints: [category] (count), [category] (count). Details: ... Recommended actions: ...'
  - Saves as data/processed/summary_pairs.jsonl
  - Format: {product_id, reviews_text, summary}
- Goal: 1500 pairs
- Track with DVC
Note: This runs on Colab or uses API calls. Budget ~$5-10 for API if using GPT-4-mini."

### Step 5.2 — Fine-tune Mistral-7B with QLoRA
Prompt: "Write notebooks/02_finetune_mistral.ipynb (designed for Google Colab):
- Install: transformers, peft, bitsandbytes, trl, datasets
- Load Mistral-7B-Instruct-v0.2 in 4-bit quantization
- Configure QLoRA: r=16, lora_alpha=32, target_modules=['q_proj','v_proj']
- Training: SFTTrainer from trl library
- Input format: '<s>[INST] Summarize the following product reviews:\n{reviews} [/INST]{summary}</s>'
- Train for 3 epochs, lr=2e-4, batch_size=4, gradient_accumulation=4
- Save adapter weights to models/mistral_finetuned/
- Log training loss curve
- Test: generate summaries for 5 unseen products, print results
- Download adapter weights for local use"

### Step 5.3 — Build LLM-as-Judge evaluation pipeline
Prompt: "Write src/evaluation/llm_judge.py that:
- Takes a product's reviews + generated summary as input
- Calls an evaluator LLM (GPT-4-mini or Claude) with scoring rubric:
  - Accuracy (1-5): Are the claims in the summary factually supported by the reviews?
  - Completeness (1-5): Does it cover the main complaint categories?
  - Actionability (1-5): Does it include clear recommended actions?
  - Conciseness (1-5): Is it appropriately concise without losing key info?
- Returns structured scores as JSON
- Also implement RAGAS metrics: faithfulness, answer_relevancy
- Stores results in PostgreSQL table: evaluations
- Batch evaluate: run on 200 test products with both base and fine-tuned model
Write tests/test_evaluation.py."

### Step 5.4 — A/B testing framework
Prompt: "Write src/evaluation/ab_test.py that:
- Loads evaluation scores for base_model and finetuned_model from PostgreSQL
- For each metric (accuracy, completeness, actionability, conciseness):
  - Computes mean and std for both models
  - Runs two-sample t-test (scipy.stats.ttest_ind)
  - Computes 95% confidence interval for the difference
  - Computes Cohen's d effect size
- Generates a summary report:
  - Table: metric | base_mean | finetuned_mean | p_value | significant? | effect_size
  - Conclusion: 'Fine-tuned model outperforms base model on X/4 metrics with statistical significance (p < 0.05)'
- Saves report as evaluation_report.md
- Logs to MLflow as artifact
Write tests/test_ab.py."

---

## PHASE 6: AGENTIC AI (Days 18-21)

### Step 6.1 — Review Analyzer Agent
Prompt: "Write src/agents/analyzer.py that:
- Takes a product_id as input
- Pulls all reviews from PostgreSQL
- Runs them through the root cause classifier
- Builds a complaint profile: {category: count, percentage, sample_reviews}
- Runs anomaly detector on recent review trends
- Returns structured AnalysisResult object
- This is a LangGraph node, not a standalone script
Use Pydantic for the data models."

### Step 6.2 — Listing Auditor Agent
Prompt: "Write src/agents/auditor.py that:
- Takes AnalysisResult + product metadata as input
- Compares product listing (title, description, features) against complaint profile
- Identifies mismatches:
  - Listing says '8-hour battery' but 60% of reviews say '4-5 hours' → MISMATCH
  - Listing doesn't mention common complaint (bluetooth issues) → MISSING_WARNING
- Uses ChromaDB to find semantically similar complaints
- Returns AuditResult with list of mismatches and severity scores
This is a LangGraph node."

### Step 6.3 — Listing Rewriter Agent
Prompt: "Write src/agents/rewriter.py that:
- Takes AuditResult as input
- Uses fine-tuned Mistral to generate:
  - Improved product title (if needed)
  - Improved description incorporating honest disclaimers
  - Suggested FAQ entries based on common complaints
- Returns RewriteResult with original vs suggested text, diff highlighted
This is a LangGraph node."

### Step 6.4 — Supervisor + LangGraph orchestration
Prompt: "Write src/agents/supervisor.py that:
- Uses LangGraph to create the agent workflow:
  - START → Analyzer → Auditor → Rewriter → HUMAN_REVIEW → END
  - If anomaly_score < threshold: START → Analyzer → END (no rewrite needed)
  - HUMAN_REVIEW is a checkpoint: shows suggestions, waits for approval
- State management: ProductAnalysisState with all intermediate results
- Error handling: if any agent fails, log error and continue with partial results
- Write src/agents/graph.py that defines the full LangGraph StateGraph
- Test: run full pipeline on 3 products, verify end-to-end output
Write tests/test_agents.py."

---

## PHASE 7: API & CI/CD (Days 22-24)

### Step 7.1 — FastAPI backend
Prompt: "Write src/api/main.py with FastAPI endpoints:
- POST /analyze/{product_id} — runs full agent pipeline, returns analysis
- GET /alerts — returns recent quality alerts from anomaly detector
- GET /alerts/{product_id} — returns alerts for specific product
- POST /classify — takes review text, returns root cause classification
- POST /search — semantic search over reviews
- GET /products/{product_id}/summary — returns pre-computed or on-demand summary
- GET /health — health check
- Add Pydantic models for all request/response schemas
- Add error handling and logging
- Swagger docs auto-generated at /docs
- CORS middleware for Streamlit frontend
Write tests/test_api.py using TestClient."

### Step 7.2 — GitHub Actions CI/CD
Prompt: "Create .github/workflows/ci.yml that:
- Triggers on: push to main, pull requests
- Steps:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies (poetry install)
  4. Run linting (flake8, black --check, isort --check)
  5. Run unit tests (pytest tests/ -v)
  6. Run model validation: load each model, run inference on 5 test inputs, assert outputs are reasonable
  7. Build Docker image
  8. (Optional) Push to Docker Hub or GitHub Container Registry
- Add badge to README"

### Step 7.3 — Evidently drift monitoring
Prompt: "Write src/mlops/drift_monitor.py that:
- Compares current week's review data against baseline (training data distribution)
- Uses Evidently to generate:
  - Data drift report (rating distribution, text length, sentiment shift)
  - Model performance report (classifier accuracy on recent labeled samples)
- Saves HTML report to reports/drift/
- If drift detected (p < 0.05 on any key feature): creates an alert in PostgreSQL
- Log drift metrics to MLflow
- Schedule: designed to run weekly (add to Makefile: make drift-report)"

---

## PHASE 8: DASHBOARD & POLISH (Days 25-30)

### Step 8.1 — Streamlit dashboard
Prompt: "Write src/dashboard/app.py with Streamlit:
Page 1: Alert Dashboard
  - Table of recent quality alerts (product name, alert type, severity, date)
  - Click an alert → see complaint breakdown chart + summary
Page 2: Product Deep Dive
  - Search/select a product
  - Show: rating trend over time (line chart), complaint category pie chart, top 5 most helpful negative reviews, executive summary, listing optimization suggestions
Page 3: Classifier Demo
  - Text input: paste any review
  - Shows: predicted root cause categories with confidence scores
Page 4: Model Performance
  - A/B test results table
  - MLflow experiment comparison
  - Drift monitoring status
- Use plotly for interactive charts
- Connect to FastAPI backend for all data
- Clean, professional UI"

### Step 8.2 — Pre-compute demo data
Prompt: "Write src/scripts/precompute_demo.py that:
- Runs the full pipeline on top 100 most-reviewed products
- Generates and stores: anomaly scores, complaint profiles, summaries, listing suggestions
- Saves to PostgreSQL so the Streamlit dashboard loads instantly
- This is what powers the live demo — users see real pre-computed results
- Also generate 5 'showcase' products with interesting anomaly stories for the demo video"

### Step 8.3 — README and documentation
Prompt: "Write a comprehensive README.md:
- Project title and one-line description
- Architecture diagram (create using mermaid or ASCII)
- Problem statement (3 sentences)
- Features (4 bullets with screenshots)
- Tech stack table
- Quick start: docker-compose up + 3 commands
- Project structure tree
- Model performance metrics table
- A/B test results summary
- Live demo link
- Demo video link
- Author and contact info
- License: MIT
Make it portfolio-ready — this is what recruiters see first."

### Step 8.4 — Demo video script
Prompt: "Write a script/outline for a 5-minute demo video:
0:00-0:30 — Problem statement (why this matters, $50B number)
0:30-1:30 — Dashboard walkthrough (alerts, product deep dive)
1:30-2:30 — Classifier demo (paste review, show classification)
2:30-3:30 — LLM summarization (show fine-tuned vs base model output)
3:30-4:30 — LangGraph agent pipeline (show full flow with terminal output)
4:30-5:00 — Architecture + tech stack summary, MLflow dashboard, close
Record with OBS or Loom. Upload to YouTube (unlisted). Link in README."

### Step 8.5 — Deploy Streamlit
Prompt: "Deploy the Streamlit dashboard:
- Create requirements.txt for Streamlit Cloud (subset of dependencies — no PyTorch/heavy models)
- Dashboard reads from pre-computed data (PostgreSQL or cached JSON files)
- The classifier runs on a small model that fits in Streamlit Cloud memory
- Deploy to Streamlit Cloud
- Get live URL
- Add URL to README and resume"

---

## PHASE 9: FINAL CHECKS (Day 30)

### Step 9.1 — Code quality
- [ ] All tests passing (pytest)
- [ ] CI/CD pipeline green
- [ ] No secrets in code (.env only)
- [ ] Black + isort formatted
- [ ] Type hints on all functions

### Step 9.2 — Documentation
- [ ] README complete with screenshots
- [ ] Architecture diagram included
- [ ] Model metrics documented
- [ ] A/B test report in repo
- [ ] Demo video recorded and linked

### Step 9.3 — Portfolio ready
- [ ] Live Streamlit URL working
- [ ] GitHub repo public and clean
- [ ] Demo video on YouTube
- [ ] Resume updated with project
- [ ] Can explain each component in 2 minutes

---

## QUICK REFERENCE: What to Say to Claude Code

For each step, your prompt to Claude Code should be:
1. "Read PROJECT_INSTRUCTIONS.md for context"
2. "We're on Step X.X of the build roadmap"
3. Paste the specific prompt from the step
4. "Test it and show me the output"

If something breaks:
- "Fix this error: [paste error]"
- "The test is failing because: [describe]"
- "This doesn't match the spec — it should do X instead of Y"

If you need to modify the plan:
- "I want to change Step X to do Y instead — update the code"
- "Skip Step X for now, we'll come back to it"
- "Add a new step between X and Y that does Z"