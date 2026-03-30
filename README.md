# Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring

End-to-end ML/AI platform that automates e-commerce product review intelligence — from anomaly detection and root cause classification to LLM-powered summarization and multi-agent listing optimization.

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
  │  Reviews (1M)   │  │  Reviews (20M) │  │  Review Embeddings│
  │  Products       │  │  Products      │  │  (384-dim vectors)│
  │  Alerts         │  │  (Full raw     │  │                   │
  │  Features       │  │   warehouse)   │  │  Semantic Search  │
  │  Evaluations    │  └────────────────┘  └───────────────────┘
  └─────────────────┘

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
PostgreSQL ◄──── Snowflake (full dataset warehouse)
    │
    ├──► spaCy NER ──► Entity Extraction (components, issues)
    ├──► Feature Pipeline ──► Rolling sentiment, velocity, negative ratio
    │
    ├──► Root Cause Classifier ──► defect / shipping / description / size / price
    ├──► Anomaly Detector ──► quality alert when sentiment spikes
    ├──► Helpfulness Predictor ──► scores review actionability
    │
    ├──► Sentence Transformers ──► ChromaDB (vector embeddings)
    │
    └──► LangGraph Agent Pipeline
              Analyzer (reviews + NER + classifier)
                  ▼
              Auditor (listing vs complaints)
                  ▼
              Rewriter (Mistral-7B generates fixes)
                  ▼
              Supervisor (human approval)
                  ▼
              FastAPI ──► Streamlit Dashboard
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Snowflake, PostgreSQL, ChromaDB, spaCy, DVC |
| ML | PyTorch (anomaly detector, root cause classifier, helpfulness predictor) |
| LLM | Mistral-7B + QLoRA, LLM-as-Judge, RAGAS |
| Agents | LangGraph (4-agent listing optimization) |
| MLOps | MLflow, GitHub Actions, Evidently AI, Docker |
| Serving | FastAPI, Streamlit |

## Data Source

**[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** — McAuley Lab, UCSD
- 571M reviews across 33 product categories (we use **Electronics** subset)
- Includes product metadata: title, description, price, features, average rating
- Free, publicly available on HuggingFace

## Quick Start

```bash
make install          # Install dependencies
docker-compose up -d  # Start PostgreSQL, ChromaDB
make test             # Run tests
```

## Author

**Bhavya Likhitha Bukka** — MS Information Systems, Northeastern University
