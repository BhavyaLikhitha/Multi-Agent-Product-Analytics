# Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-green.svg)](https://github.com/langchain-ai/langgraph)
[![Mistral-7B](https://img.shields.io/badge/Mistral--7B-QLoRA-orange.svg)](https://mistral.ai)

End-to-end ML/AI platform that automates e-commerce product review intelligence — from anomaly detection and root cause classification to LLM-powered summarization and multi-agent listing optimization.

---

## Problem Statement

E-commerce companies lose **$50B+/year** on product returns, many caused by quality issues that appear in reviews long before action is taken. Product managers manually sift through thousands of reviews — slow, inconsistent, and unscalable.

This platform **automates the entire pipeline**: detect quality anomalies in real-time, classify *why* customers are unhappy, generate executive summaries, and produce improved product listings — all with human-in-the-loop approval.

---

## Key Results

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| Root Cause Classifier (DistilBERT) | Macro-F1 | > 0.70 | **0.7339** |
| Anomaly Detector (Autoencoder) | Quality Alerts | — | **21,310 detected** |
| Helpfulness Predictor (Neural Net) | MAE | < 2.0 | **1.46** |
| Fine-tuned Mistral-7B (QLoRA) | LLM Judge Score | > 3.5/5 | **3.90/5** |
| Semantic Search (ChromaDB) | Embeddings | — | **50,000 reviews indexed** |

**Data scale:** 500K reviews + 1.6M products from Amazon Electronics (pipeline supports 20M+)

---

## Features

- **Quality Alert System** — PyTorch autoencoder detects anomalous review sentiment spikes per product
- **Root Cause Classifier** — DistilBERT multi-label classifier identifies *why* customers complain (defect, shipping, description, size, price)
- **Executive Summarizer** — Mistral-7B fine-tuned with QLoRA generates structured complaint summaries per product
- **Listing Optimizer** — LangGraph 4-agent pipeline (Analyzer → Auditor → Rewriter → Supervisor) rewrites misleading listings
- **Semantic Search** — Natural language search over 50K review embeddings using sentence-transformers + ChromaDB
- **NER Extraction** — Custom regex NER with 45+ component patterns and 120+ issue patterns
- **Drift Monitoring** — Evidently AI tracks feature distribution shifts to flag model degradation
- **Interactive Dashboard** — 5-page Streamlit app: Alerts, Product Deep Dive, Classifier Demo, Semantic Search, Model Performance

---

## Architecture

```
                         ┌──────────────────────────────────┐
                         │        Streamlit Dashboard        │
                         │  Alerts | Deep Dive | Classifier  │
                         └──────────────┬───────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    FastAPI Backend  │
                              │  /analyze /alerts   │
                              │  /classify /search   │
                              └────┬──────────┬────┘
                                   │          │
                    ┌──────────────▼──┐  ┌────▼──────────────────┐
                    │   LangGraph      │  │    ML Models (PyTorch) │
                    │   Agent Pipeline │  │                        │
                    │                  │  │  Root Cause Classifier  │
                    │  ┌─Analyzer──┐   │  │  (DistilBERT multi-label)│
                    │  │  Reviews  │   │  │                        │
                    │  │  + NER    │   │  │  Anomaly Detector      │
                    │  └────┬──────┘   │  │  (Autoencoder)         │
                    │  ┌────▼──────┐   │  │                        │
                    │  │  Auditor  │   │  │  Helpfulness Predictor │
                    │  │  Listing  │   │  │  (Neural Network)      │
                    │  │  vs Reviews│  │  └────────────────────────┘
                    │  └────┬──────┘   │
                    │  ┌────▼──────┐   │
                    │  │ Rewriter  │   │       ┌──────────────────┐
                    │  │ Mistral-7B│───│───────│  Mistral-7B      │
                    │  └────┬──────┘   │       │  QLoRA Fine-tuned │
                    │  ┌────▼──────┐   │       │  (Summarization)  │
                    │  │Supervisor │   │       └──────────────────┘
                    │  │Human Review│  │
                    │  └───────────┘   │
                    └──────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
  ┌───────▼────────┐  ┌────────▼───────┐  ┌─────────▼────────┐
  │  PostgreSQL     │  │   Snowflake    │  │    ChromaDB       │
  │  (Docker)       │  │   (Cloud)      │  │    (Docker)       │
  │                 │  │                │  │                   │
  │  Reviews (500K) │  │  Reviews (500K)│  │  Review Embeddings│
  │  Products (1.6M)│  │  Products(1.6M)│  │  (384-dim vectors)│
  │  Alerts         │  │  (Warehouse)   │  │                   │
  │  Features       │  └────────────────┘  │  Semantic Search  │
  └─────────────────┘                      └───────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │                        MLOps Layer                           │
  │  MLflow (experiment tracking) │ Evidently (drift monitoring) │
  │  GitHub Actions (CI/CD)       │ DVC (pipeline versioning)    │
  └──────────────────────────────────────────────────────────────┘
```

## Data Flow

```
HuggingFace (Amazon Reviews 2023, Electronics)
    │
    ▼ streaming (no local storage)
PostgreSQL ──────► Snowflake (warehouse)
    │
    ├──► Regex NER ──► Entity Extraction (components, issues)
    ├──► Feature Pipeline ──► Rolling sentiment, velocity, negative ratio
    │
    ├──► Root Cause Classifier ──► defect / shipping / description / size / price
    ├──► Anomaly Detector ──► quality alert when sentiment spikes
    ├──► Helpfulness Predictor ──► scores review actionability
    │
    ├──► Sentence Transformers ──► ChromaDB (vector embeddings)
    │
    └──► LangGraph Agent Pipeline
              Analyzer ──► Auditor ──► Rewriter ──► Supervisor
                                                       │
              FastAPI ◄────────────────────────────────┘
                  │
              Streamlit Dashboard
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Snowflake, PostgreSQL, ChromaDB, spaCy, DVC |
| ML | PyTorch (3 models), MLflow for experiment tracking |
| LLM | Mistral-7B + QLoRA, LLM-as-Judge, Claude Sonnet (training data) |
| Agents | LangGraph (4-agent listing optimization) |
| MLOps | GitHub Actions CI/CD, Evidently AI drift monitoring, Docker |
| Serving | FastAPI backend, Streamlit dashboard |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Bhavyalikhitha/Multi-Agent-Product-Analytics.git
cd Multi-Agent-Product-Analytics
poetry install

# 2. Set up environment
cp .env.example .env
# Edit .env with your credentials (Snowflake, Groq API key)

# 3. Start databases
docker-compose up -d    # PostgreSQL + ChromaDB

# 4. Load data
poetry run python src/data/download.py        # Stream 500K reviews into PostgreSQL
poetry run python src/data/load_snowflake.py   # Load into Snowflake warehouse

# 5. Run feature pipeline
poetry run python src/features/feature_pipeline.py

# 6. Train models
poetry run python src/models/anomaly_detector.py
poetry run python src/models/helpfulness_predictor.py
poetry run python src/models/root_cause_classifier.py

# 7. Generate embeddings
poetry run python src/features/generate_embeddings.py

# 8. Launch dashboard
poetry run streamlit run src/dashboard/app.py
# Open http://localhost:8501
```

---

## Project Structure

```
src/
├── data/               # Data ingestion and labeling
│   ├── download.py           # Stream HuggingFace → PostgreSQL
│   ├── load_snowflake.py     # PostgreSQL → Snowflake warehouse
│   ├── load_postgres.py      # Table schemas and indexes
│   ├── create_labels.py      # Groq LLM-assisted root cause labeling (3.5K reviews)
│   ├── create_summary_pairs.py  # Claude Sonnet training pair generation (400 pairs)
│   └── label_shipping.py     # Targeted shipping label augmentation
│
├── features/           # Feature engineering
│   ├── ner_extractor.py      # Regex NER (45 components, 120+ issues)
│   ├── feature_pipeline.py   # Rolling sentiment, velocity, negative ratio
│   └── generate_embeddings.py # Sentence-transformers → ChromaDB
│
├── models/             # PyTorch models
│   ├── root_cause_classifier.py  # DistilBERT multi-label (F1=0.7339)
│   ├── anomaly_detector.py       # Autoencoder (21K alerts)
│   ├── helpfulness_predictor.py  # Feedforward NN (MAE=1.46)
│   └── eval_classifier.py       # Threshold tuning + evaluation
│
├── agents/             # LangGraph multi-agent pipeline
│   ├── analyzer.py     # Review analysis + NER + complaint profile
│   ├── auditor.py      # Listing vs complaints mismatch detection
│   ├── rewriter.py     # LLM-powered listing rewriter
│   ├── supervisor.py   # Rule-based approval (human-in-the-loop)
│   └── graph.py        # LangGraph orchestration with conditional routing
│
├── evaluation/         # LLM evaluation framework
│   ├── llm_judge.py    # LLM-as-Judge (4 criteria, 1-5 scale)
│   └── ab_test.py      # Statistical A/B testing (t-test)
│
├── api/                # FastAPI backend
│   ├── main.py         # 6 endpoints (health, alerts, classify, search, analyze, products)
│   └── semantic_search.py  # ChromaDB cosine similarity search
│
├── mlops/              # MLOps tooling
│   └── drift_monitor.py    # Evidently AI feature drift detection
│
└── dashboard/          # Streamlit frontend
    └── app.py          # 5-page dashboard (Alerts, Deep Dive, Classifier, Search, Metrics)

notebooks/
├── 01_eda.ipynb                # Exploratory data analysis (8 plots)
├── 02_finetune_mistral.ipynb   # QLoRA fine-tuning on Colab T4
└── 03_mistral_inference.ipynb  # Batch inference with fine-tuned model
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Walkthrough](docs/WALKTHROUGH.md) | Step-by-step interview prep — explains every phase with "how to explain" scripts |
| [Decisions](docs/decisions.md) | 13 architecture decisions with reasoning, tradeoffs, and interview-ready answers |
| [Progress](docs/progress.md) | Build timeline, session logs, metrics achieved |
| [Roadmap](docs/roadmap.md) | Original project plan |

---

## Data Source

**[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** — McAuley Lab, UCSD
- 571M reviews across 33 product categories (we use **Electronics** subset)
- Includes product metadata: title, description, price, features, average rating
- Free, publicly available on HuggingFace

---

## Author

**Bhavya Likhitha Bukka** — MS Information Systems, Northeastern University
