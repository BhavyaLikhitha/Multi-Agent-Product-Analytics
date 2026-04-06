# Project Instructions — Bhavya Likhitha Bukka

## Who I Am
- MS in Information Systems, Northeastern University (3.83 GPA), graduating August 2026
- Based in Boston. Need job before graduation. Need visa sponsorship (OPT → H1B).
- Experience: Fidelity co-op (Data Analyst, 6 months USA), I Ray IT (Software Data Engineer, 1.5 years India), Devtown (Data Scientist Intern, 8 months India)
- Targeting: Data Engineer (40%), Data Scientist GenAI/LLM (30%), AI Engineer (20%), Data Analyst (10% backup)

## My 3 Resume Projects
1. **Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring** ← THIS IS THE NEW PROJECT (building now)
2. **PE OrgAIR Platform** — RAG + agentic AI for private equity investment analysis (in progress, week 9 of 14)
3. **Last-Mile Fulfillment Optimization** — Data engineering + ML for supply chain (complete)

Three different domains (e-commerce, finance, supply chain). Three different core skills (ML+LLM+MLOps, RAG+agents, DE+ML). Zero overlap.

---

## The New Project: Multi-Agent Product Analytics with LLM Intelligence and Quality Monitoring

### Problem It Solves
E-commerce companies lose $50B+/year on returns. Product teams can't process millions of reviews fast enough to catch quality issues, understand root causes, or fix misleading listings. This platform automates that entire pipeline.

### 4 Outputs
1. **Quality Alert System** — Detects anomalous review sentiment spikes per product (PyTorch anomaly detection)
2. **Root Cause Classifier** — Labels every review: defect, shipping, description, size, price (PyTorch multi-label transformer)
3. **Executive Summarizer** — Generates structured product summaries with fine-tuned Mistral-7B (QLoRA), evaluated by LLM-as-Judge + RAGAS, A/B tested against base model
4. **Listing Optimizer Agent** — LangGraph multi-agent system: Analyzer → Auditor → Rewriter → Supervisor with human-in-the-loop

### Tech Stack
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data | Amazon Reviews 2023 (HuggingFace, 571M reviews, free) | Source data |
| Data | Snowflake | Data warehouse for raw review data |
| Data | PostgreSQL | Application database (alerts, tracking) |
| Data | ChromaDB | Vector store for review embeddings |
| Data | spaCy NER | Entity extraction from review text |
| Data | DVC | Data versioning |
| ML | PyTorch — Anomaly Detector (autoencoder) | Detects review sentiment spikes |
| ML | PyTorch — Root Cause Classifier (DistilBERT multi-label) | Classifies complaint categories |
| ML | PyTorch — Helpfulness Predictor (neural net) | Scores review actionability |
| LLM | Mistral-7B + QLoRA fine-tuning (Google Colab) | Generates executive product summaries |
| LLM | LLM-as-Judge + RAGAS | Evaluates summary quality |
| LLM | A/B Testing (scipy.stats) | Proves fine-tuned > base model with p < 0.05 |
| MLOps | MLflow | Experiment tracking + model registry |
| MLOps | GitHub Actions | CI/CD pipeline |
| MLOps | Evidently AI | Drift monitoring |
| MLOps | Docker Compose | Containerization |
| Agents | LangGraph | 4-agent listing optimization system |
| Serving | FastAPI | REST API backend |
| Serving | Streamlit | Product manager dashboard |

### Data Source
- McAuley Lab Amazon Reviews 2023 — free on HuggingFace
- 571M reviews, 33 categories, with product metadata
- Start with Electronics subset (~20M reviews)
- Must create: root cause labels (LLM-assisted + manual, 2-3 days) and summary training pairs (1500 curated, 3-4 days)

### 6-Week Build Plan (full-time, started March 17)
- **Week 1 (Mar 17-23):** Repo setup, data pipeline, Snowflake + PostgreSQL, spaCy NER, DVC, EDA
- **Week 2 (Mar 24-30):** Root cause labels + PyTorch classifier + anomaly detector + MLflow
- **Week 3 (Apr 1-6):** Helpfulness predictor + ChromaDB embeddings + FastAPI endpoints
- **Week 4 (Apr 7-13):** LLM fine-tuning (QLoRA on Colab) + LLM-as-Judge eval + A/B testing
- **Week 5 (Apr 14-20):** LangGraph agents + GitHub Actions CI/CD + Evidently + Docker
- **Week 6 (Apr 21-27):** Streamlit dashboard + README + demo video + deploy

### Cost: $0 - $10
Everything is open-source or free tier.

---

## How to Work With Me (for Claude)

### Communication Style
- Be concise. Don't give 10-page answers when 1 paragraph works.
- Be honest. Tell me when something won't work or when I'm wrong.
- Be specific. No vague advice — give exact tools, exact steps, exact code.
- Challenge me back when I push back — I respect data-driven disagreements.

### What I Need Help With (in priority order)
1. Building this project week by week — code, architecture, debugging
2. Resume tailoring for specific job applications
3. Job search strategy — which companies, referral outreach templates
4. Interview prep — DS, DE, AI Engineer questions

### What NOT to Do
- Don't give generic career advice. I've heard it all.
- Don't suggest tools I haven't agreed to. Stick to the tech stack above.
- Don't overexplain things I already know (SQL, Python, Airflow, Snowflake, dbt, Power BI).
- Don't repeat context I've already given — reference this file instead.
- Don't give unnecessarily long responses. If I ask a yes/no question, give me yes/no first, then explain if needed.

### Context to Remember
- I'm short on money. Everything must be free or near-free.
- I'm on F1 visa. Need OPT/H1B sponsorship. This affects company targeting.
- My MS is in Information Systems, not CS/ML. This matters for resume screening.
- My strongest experience title match: "Data Scientist" (Devtown) and "Data Engineer" (I Ray IT).
- Meta/Google DS interviews are stats-heavy — I need separate prep for those.
- Capital One, Deloitte, Amazon, Microsoft DS roles ask for LLM/PyTorch/agentic — my project directly matches.
- I should be applying to jobs IN PARALLEL with building this project, not waiting until it's done.

### GitHub Repos
- Last-Mile Fulfillment: https://github.com/BhavyaLikhitha/Last-Mile-Fulfilment-Optimization
- PE OrgAIR: https://github.com/BigDataIA-Spring26-Team-5/PE_OrgAIR_Platform_RAG
- Airbnb Pipeline: https://github.com/BhavyaLikhitha/Airbnb-Data-Engineering-and-Analytics-Pipeline
- Amazon Sales: https://github.com/BhavyaLikhitha/Amazon-Sales-Analysis-SQL
- Portfolio Optimization ML: https://github.com/BhavyaLikhitha/Portfolio-Optimization-and-Market-Performance-Analysis-using-ML
- IMDB Analysis: https://github.com/DAMG7370-FALL2025-GROUP9/IMDB_analysis
- UBER Rides: https://github.com/BhavyaLikhitha/UBER-Rides
- Phone Call Centre: https://github.com/BhavyaLikhitha/Phone-Call-Centre-Recordings-Analysis
- Bank Loans: https://github.com/BhavyaLikhitha/Bank-Loans-Financial-Analysis
- Costco Sales: https://github.com/BhavyaLikhitha/Costco-Sales-Analysis

### Resume PDF
Available in uploads as bhavya_meta_resume.pdf

### Key Documents Created
- Job Search Strategy Guide (bhavya_job_search_strategy.docx)
- Project Spec (final_project_spec.docx)
- These project instructions (PROJECT_INSTRUCTIONS.md)
- Build Roadmap with 30 step-by-step prompts (BUILD_ROADMAP.md)
- Progress Tracker — update after every session (PROGRESS.md)
- Decisions & Tradeoffs Log — every design choice documented (DECISIONS.md)

### MANDATORY RULE FOR CLAUDE CODE: Updating DECISIONS.md
Whenever Claude Code changes, overrides, or deviates from ANY decision in DECISIONS.md — or makes a NEW technical decision (choosing a library, changing architecture, picking a different approach than planned) — it MUST:
1. Immediately tell me: "I'm making/changing a decision. Here's what and why."
2. Update DECISIONS.md with: what was decided, what alternatives existed, why this choice, what the tradeoff is.
3. If it contradicts an existing decision in DECISIONS.md, update the old entry with "[REVISED]" and add the new reasoning below it.
This is non-negotiable. Every design choice must be documented for interview prep.

### How to Start a New Claude Code Session
1. Upload: PROJECT_INSTRUCTIONS.md + BUILD_ROADMAP.md + PROGRESS.md + DECISIONS.md
2. Say: "Read all 4 files. Check PROGRESS.md for where we left off. Continue from the next incomplete step. If you change any decision or make a new one, update DECISIONS.md immediately."
3. After session ends, update PROGRESS.md with: steps completed, issues, decisions made, next step.

### How to Hand Off to a New Chat
Upload all 4 files + say: "I'm continuing a project. Read all files for full context. PROGRESS.md shows where I am. DECISIONS.md has all tradeoffs. If you change any decision, update DECISIONS.md immediately. Continue from the next incomplete step in BUILD_ROADMAP.md."

