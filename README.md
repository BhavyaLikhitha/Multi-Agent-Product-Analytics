# Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring

End-to-end ML/AI platform that automates e-commerce product review intelligence — from anomaly detection and root cause classification to LLM-powered summarization and multi-agent listing optimization.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Snowflake, PostgreSQL, ChromaDB, spaCy, DVC |
| ML | PyTorch (anomaly detector, root cause classifier, helpfulness predictor) |
| LLM | Mistral-7B + QLoRA, LLM-as-Judge, RAGAS |
| Agents | LangGraph (4-agent listing optimization) |
| MLOps | MLflow, GitHub Actions, Evidently AI, Docker |
| Serving | FastAPI, Streamlit |

## Quick Start

```bash
make install          # Install dependencies
docker-compose up -d  # Start PostgreSQL, ChromaDB, MLflow
make test             # Run tests
```

## Author

**Bhavya Likhitha Bukka** 