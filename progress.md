# Progress Tracker — Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring

## Last Updated: 2026-04-03

---

## Current Status
- **Current Phase:** Phase 5 complete, Phase 6 code done, Phase 7 API done
- **Current Step:** Remaining: Streamlit dashboard, GitHub Actions, Evidently drift, final polish
- **Blockers:** None
- **Next Action:** Build Streamlit dashboard (Phase 8)

---

## Phase Completion

| Phase | Description | Status | Date Started | Date Completed |
|-------|-------------|--------|-------------|----------------|
| Phase 1 | Project Setup | ✅ Complete | 2026-03-30 | 2026-03-30 |
| Phase 2 | Data Pipeline | ✅ Complete | 2026-03-30 | 2026-03-31 |
| Phase 3 | ML Models (PyTorch) | ✅ Complete | 2026-03-31 | 2026-04-01 |
| Phase 4 | Embeddings & Vector Search | ✅ Complete | 2026-04-01 | 2026-04-01 |
| Phase 5 | LLM Fine-tuning & Evaluation | ✅ Complete | 2026-04-01 | 2026-04-03 |
| Phase 6 | Agentic AI (LangGraph) | ✅ Complete | 2026-04-02 | 2026-04-02 |
| Phase 7 | API & CI/CD | 🟡 In Progress | 2026-04-02 | — |
| Phase 8 | Dashboard & Polish | ⬜ Not Started | — | — |
| Phase 9 | Final Checks | ⬜ Not Started | — | — |

---

## Step-by-Step Completion

### Phase 1: Project Setup
- [x] 1.1 — Initialize repo
- [x] 1.2 — Docker foundation
- [x] 1.3 — DVC setup

### Phase 2: Data Pipeline
- [x] 2.1 — Stream 500K reviews + 1.6M products into PostgreSQL (done)
- [x] 2.2 — Load into Snowflake (500K reviews + 1.6M products loaded)
- [x] 2.3 — PostgreSQL tables + indexes (done)
- [x] 2.4 — spaCy NER pipeline (120+ issue patterns, 45+ components, 9/9 tests passing, running on full data)
- [x] 2.5 — Feature engineering pipeline (running on 500K reviews)
- [x] 2.6 — EDA notebook (executed, validated data + NER output)

### Phase 3: ML Models
- [x] 3.1 — Create root cause labels (3.5K reviews: 3K general + 500 shipping-targeted, via Groq)
- [x] 3.2 — Root cause classifier (DistilBERT, test F1=0.7339, TARGET PASSED)
- [x] 3.3 — Anomaly detector trained (Autoencoder, 21,310 alerts, loss 0.087)
- [x] 3.4 — Helpfulness predictor trained (MAE=1.46, TARGET PASSED)
- [x] 3.5 — MLflow experiment tracking (local file-based, 3 experiments logged)

### Phase 4: Embeddings & Vector Search
- [x] 4.1 — Generate review embeddings (50,000 reviews in ChromaDB, all-MiniLM-L6-v2, 384-dim)
- [x] 4.2 — Semantic search endpoint (src/api/semantic_search.py, with rating filters)

### Phase 5: LLM Fine-tuning & Evaluation
- [x] 5.1 — Curate summary training data (400 pairs via Claude Sonnet, upgraded from 200 Llama 8B)
- [x] 5.2 — Fine-tune Mistral-7B with QLoRA (2 epochs on Colab T4, r=8, q/v_proj)
- [x] 5.3 — LLM-as-Judge evaluation (Fine-tuned: 3.90/5, Base: 3.94/5)
- [x] 5.4 — A/B testing (no significant difference — documented as valid experimental result)

### Phase 6: Agentic AI
- [x] 6.1 — Review Analyzer Agent (NER + complaint profile)
- [x] 6.2 — Listing Auditor Agent (mismatch detection)
- [x] 6.3 — Listing Rewriter Agent (Groq-powered rewriting)
- [x] 6.4 — Supervisor + LangGraph orchestration (conditional routing, tested)

### Phase 7: API & CI/CD
- [x] 7.1 — FastAPI backend (6 endpoints: health, alerts, classify, search, analyze, products)
- [ ] 7.2 — GitHub Actions CI/CD
- [ ] 7.3 — Evidently drift monitoring

### Phase 8: Dashboard & Polish
- [ ] 8.1 — Streamlit dashboard
- [ ] 8.2 — Pre-compute demo data
- [ ] 8.3 — README and documentation
- [ ] 8.4 — Demo video script
- [ ] 8.5 — Deploy Streamlit

### Phase 9: Final Checks
- [ ] 9.1 — Code quality
- [ ] 9.2 — Documentation
- [ ] 9.3 — Portfolio ready

---

## Session Log

### Session 1 — 2026-03-30
**Steps completed:** 1.1, 1.2, 1.3, 2.1-2.6 (all code written)
**Issues encountered:** dvc.lock was in .gitignore — removed. spaCy EntityRuler needed LOWER token matching for case-insensitivity.
**Decisions made:** T1: ChromaDB Cloud instead of Docker
**Next session starts at:** Run data download pipeline

### Session 2 — 2026-03-30 (evening)
**Steps completed:** 2.1 (500K reviews streamed), 2.2 (Snowflake loaded), 2.3 (PostgreSQL verified), 2.6 (EDA notebook run)
**Issues encountered:**
- Disk space: couldn't download full dataset as parquet, switched to streaming directly into PostgreSQL
- HuggingFace metadata had nested `images` column causing pyarrow cast error — dropped complex columns before streaming
- Snowflake: REVIEWS table didn't exist on first run — added CREATE TABLE IF NOT EXISTS
- Snowflake: PostgreSQL `id` column not in Snowflake schema — dropped before insert
**Decisions made:**
- T1 [REVISED]: ChromaDB back to Docker ($4 cloud credit insufficient for 500K embeddings)
- T2: Stream directly into PostgreSQL, skip local parquet files (disk constraint)
- T3: 500K review subset for development (scales to 20M)
**Next session starts at:** Run NER + feature pipeline on 500K reviews, then Phase 3 (ML models)

### Session 3 — 2026-03-31 / 2026-04-01
**Steps completed:** 2.4 (NER run on 500K), 2.5 (features stored: 486K rows), 3.1 (3K labels via Groq), 3.3 (anomaly detector trained), 3.4 (helpfulness predictor trained, MAE=1.46 PASS), 3.2 (classifier first run F1=0.67, retraining)
**Issues encountered:**
- spaCy NER took 6+ hours on 500K reviews — switched to pre-compiled regex (2 min)
- Groq free tier daily token limit (500K tokens) — hit at 1400 reviews, script didn't save progress, lost work
- Fixed: added incremental saving every 50 reviews with resume support
- PyTorch DLL error on Windows — reinstalled CPU-only torch
- MLflow connection refused — switched from server to local file-based tracking (file:./mlruns)
- Classifier missed 0.70 target (0.67) — shipping only 22 test samples dragging macro F1 down
**Decisions made:**
- T4: Regex NER for bulk, spaCy for demos (100x faster)
- T5: Groq free tier for labeling (saves budget for Phase 5)
- T6: 3000 labeled samples (enough for 5-category classifier)
- T7: Class weights + threshold tuning to fix rare category performance
**Next session starts at:** Verify classifier F1 > 0.70, run embeddings pipeline

### Session 4 — 2026-04-01 / 2026-04-02
**Steps completed:** 3.2 (classifier PASSED F1=0.7339), 3.5 (MLflow logged), 4.1 (50K embeddings), 4.2 (semantic search), 5.1 (400 Claude Sonnet training pairs), 5.2 (Mistral QLoRA fine-tuned on Colab), 6.1-6.4 (all agents + graph), 7.1 (FastAPI)
**Issues encountered:**
- Classifier first run F1=0.67 — added 500 shipping-targeted labels + class weights → F1=0.7339 PASS
- Groq Llama 8B training data produced low quality summaries — switched to Claude Sonnet ($4.20)
- Colab T4 OOM after 5 hours — reduced config (r=8, batch=1, gradient checkpointing)
- Colab GPU quota exhaustion — used new Google account
- torch DLL error on Windows after every poetry add — repeated CPU reinstall needed
**Decisions made:**
- T9: Extra shipping labels (171→344) to fix class imbalance
- T10: 50K embeddings, negative-first priority
- T11: Claude Sonnet for training data (quality >> quantity)
- T12: Lean QLoRA config (r=8, q/v_proj, 2 epochs) to fit T4 memory

### Session 5 — 2026-04-03
**Steps completed:** 5.3 (LLM-as-Judge: fine-tuned 3.90/5, base 3.94/5), 5.4 (A/B test: no significant difference, p=0.72)
**Issues encountered:**
- Fine-tuned model scored 3.90 vs base 3.94 — no significant improvement
- Groq rate limit hit during base evaluation — lost 360 rows when cleanup script accidentally deleted them
- Judge bias: Llama 8B judging favors its own output style
**Decisions made:**
- Document A/B result honestly — fine-tuning pipeline is correct, results show no significant improvement with 400 pairs
- Interview talking point: "Pipeline works, experiment didn't show improvement — in production would use 2000+ human-verified pairs"
**Next session starts at:** Streamlit dashboard, GitHub Actions, Evidently drift, final polish

<!-- Copy this template for each new session:

### Session N — [DATE]
**Steps completed:**
**Issues encountered:**
**Decisions made:**
**Next session starts at:**

-->

---

## Metrics Achieved

| Model | Metric | Target | Actual | Status |
|-------|--------|--------|--------|--------|
| Root Cause Classifier | Macro-F1 | > 0.70 | 0.7339 | PASS |
| Anomaly Detector | Threshold@95th | — | 0.213 (21,310 alerts) | PASS |
| Helpfulness Predictor | MAE | < 2.0 | 1.46 | PASS |
| Fine-tuned Mistral vs Base | A/B p-value | < 0.05 | 0.72 | No significant diff |
| Fine-tuned Mistral | Judge Avg Score | > 3.5/5 | 3.90 | PASS |
| Base (Groq Llama) | Judge Avg Score | — | 3.94 | — |