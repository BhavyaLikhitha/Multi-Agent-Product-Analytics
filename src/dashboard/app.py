"""Streamlit dashboard for Product Intelligence platform.

Pages:
1. Quality Alerts — live alerts from anomaly detector
2. Product Deep Dive — select product, see reviews + NER + classifier
3. Classifier Demo — paste review text, get root cause labels
4. Semantic Search — natural language search over reviews
5. Model Performance — metrics summary
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get(
        "POSTGRES_DB", "product_intelligence"
    )
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(
        f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    )


engine = get_engine()

st.set_page_config(
    page_title="Product Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("Product Intelligence Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Quality Alerts",
        "Product Deep Dive",
        "Classifier Demo",
        "Semantic Search",
        "Model Performance",
    ],
)


# ─── Page 1: Quality Alerts ───────────────────────────
if page == "Quality Alerts":
    st.header("Quality Alerts")
    st.markdown(
        "Products flagged by the anomaly detection model."
    )

    severity = st.selectbox(
        "Filter by severity",
        ["All", "critical", "warning"],
    )

    query = "SELECT * FROM alerts ORDER BY detected_at DESC LIMIT 100"
    if severity != "All":
        query = (
            f"SELECT * FROM alerts "
            f"WHERE severity = '{severity}' "
            f"ORDER BY detected_at DESC LIMIT 100"
        )

    try:
        alerts = pd.read_sql(text(query), engine)
        if alerts.empty:
            st.info("No alerts found.")
        else:
            st.metric("Total Alerts", len(alerts))

            col1, col2 = st.columns(2)
            with col1:
                critical = len(
                    alerts[alerts["severity"] == "critical"]
                )
                st.metric("Critical", critical)
            with col2:
                warning = len(
                    alerts[alerts["severity"] == "warning"]
                )
                st.metric("Warning", warning)

            st.dataframe(
                alerts[
                    [
                        "product_id",
                        "alert_type",
                        "severity",
                        "detected_at",
                        "details",
                    ]
                ],
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Could not load alerts: {e}")


# ─── Page 2: Product Deep Dive ────────────────────────
elif page == "Product Deep Dive":
    st.header("Product Deep Dive")

    top_products = pd.read_sql(
        text(
            """
            SELECT r.asin, p.title,
                   COUNT(*) as review_count,
                   ROUND(AVG(r.rating)::numeric, 2)
                       as avg_rating
            FROM reviews r
            LEFT JOIN products p
                ON r.parent_asin = p.parent_asin
            GROUP BY r.asin, p.title
            ORDER BY review_count DESC
            LIMIT 50
            """
        ),
        engine,
    )

    selected = st.selectbox(
        "Select a product",
        top_products["asin"].tolist(),
        format_func=lambda x: (
            f"{x} — "
            f"{top_products[top_products['asin']==x]['title'].values[0] or 'Unknown'}"
            f" ({top_products[top_products['asin']==x]['review_count'].values[0]} reviews)"
        ),
    )

    if selected:
        info = top_products[
            top_products["asin"] == selected
        ].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Reviews", info["review_count"])
        col2.metric("Avg Rating", info["avg_rating"])
        col3.metric(
            "Product", str(info["title"])[:50]
        )

        reviews = pd.read_sql(
            text(
                "SELECT rating, title, text, "
                "helpful_votes, timestamp "
                "FROM reviews WHERE asin = :asin "
                "ORDER BY helpful_votes DESC "
                "LIMIT 20"
            ),
            engine,
            params={"asin": selected},
        )

        st.subheader("Top Reviews")
        for _, row in reviews.iterrows():
            with st.expander(
                f"{'⭐' * int(row['rating'])} "
                f"{row['title'] or 'No title'}"
            ):
                st.write(row["text"])
                st.caption(
                    f"Helpful votes: "
                    f"{row['helpful_votes']}"
                )

        # NER analysis
        st.subheader("NER Analysis")
        try:
            from src.features.ner_extractor import (
                extract_batch_fast,
            )

            texts = reviews["text"].fillna("").tolist()
            ner_results = extract_batch_fast(texts)

            from collections import Counter

            all_comps = Counter()
            all_issues = Counter()
            for ner in ner_results:
                for c in ner["components"]:
                    all_comps[c] += 1
                for i in ner["issues"]:
                    all_issues[i] += 1

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Components**")
                for comp, cnt in all_comps.most_common(10):
                    st.write(f"- {comp}: {cnt}")
            with col2:
                st.markdown("**Top Issues**")
                for iss, cnt in all_issues.most_common(10):
                    st.write(f"- {iss}: {cnt}")
        except Exception as e:
            st.warning(f"NER analysis failed: {e}")


# ─── Page 3: Classifier Demo ─────────────────────────
elif page == "Classifier Demo":
    st.header("Root Cause Classifier Demo")
    st.markdown(
        "Paste any review text to classify its root cause."
    )

    review_text = st.text_area(
        "Enter review text",
        height=150,
        placeholder=(
            "e.g., The battery died after 2 weeks "
            "and the screen is cracked..."
        ),
    )

    if st.button("Classify") and review_text:
        try:
            from src.features.ner_extractor import (
                extract_fast,
            )

            ner = extract_fast(review_text)

            st.subheader("NER Entities")
            col1, col2 = st.columns(2)
            col1.write(
                f"**Components:** {', '.join(ner['components']) or 'None'}"
            )
            col2.write(
                f"**Issues:** {', '.join(ner['issues']) or 'None'}"
            )

            # Simple rule-based scoring
            categories = {
                "defect": [
                    "broken", "defective", "dead",
                    "stopped working", "malfunction",
                ],
                "shipping": [
                    "arrived damaged", "missing parts",
                    "wrong item", "late delivery",
                ],
                "description": [
                    "misleading", "not as described",
                    "false advertising",
                ],
                "size": [
                    "too small", "too big",
                    "doesn't fit", "wrong size",
                ],
                "price": [
                    "waste of money", "overpriced",
                    "rip off", "scam",
                ],
            }

            text_lower = review_text.lower()
            scores = {}
            for cat, keywords in categories.items():
                score = sum(
                    1 for k in keywords if k in text_lower
                )
                scores[cat] = min(1.0, score * 0.3)

            st.subheader("Root Cause Classification")
            for cat, score in scores.items():
                st.progress(
                    score,
                    text=f"{cat}: {score:.0%}",
                )

        except Exception as e:
            st.error(f"Classification failed: {e}")


# ─── Page 4: Semantic Search ─────────────────────────
elif page == "Semantic Search":
    st.header("Semantic Search")
    st.markdown(
        "Search reviews by meaning, not just keywords."
    )

    query = st.text_input(
        "Search query",
        placeholder="e.g., bluetooth keeps disconnecting",
    )

    max_rating = st.slider(
        "Max rating filter", 1.0, 5.0, 3.0, 0.5
    )

    if st.button("Search") and query:
        try:
            from src.api.semantic_search import (
                search_reviews,
            )

            results = search_reviews(
                query=query,
                n_results=10,
                max_rating=max_rating,
            )

            if not results:
                st.info("No matching reviews found.")
            else:
                for r in results:
                    with st.expander(
                        f"{'⭐' * int(r['rating'])} "
                        f"{r['title']} "
                        f"(similarity: "
                        f"{1 - r['distance']:.2f})"
                    ):
                        st.write(r["text"])
                        st.caption(f"ASIN: {r['asin']}")
        except Exception as e:
            st.error(f"Search failed: {e}")


# ─── Page 5: Model Performance ───────────────────────
elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model Metrics")

    metrics = {
        "Root Cause Classifier": {
            "Metric": "Macro-F1",
            "Target": "> 0.70",
            "Actual": "0.7339",
            "Status": "PASS",
        },
        "Anomaly Detector": {
            "Metric": "Threshold@95th",
            "Target": "—",
            "Actual": "0.213 (21,310 alerts)",
            "Status": "PASS",
        },
        "Helpfulness Predictor": {
            "Metric": "MAE",
            "Target": "< 2.0",
            "Actual": "1.46",
            "Status": "PASS",
        },
        "Fine-tuned Mistral": {
            "Metric": "Judge Avg",
            "Target": "> 3.5/5",
            "Actual": "3.90/5",
            "Status": "PASS",
        },
        "A/B Test (base vs tuned)": {
            "Metric": "p-value",
            "Target": "< 0.05",
            "Actual": "0.72",
            "Status": "No sig. diff",
        },
    }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "Model"
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Classifier Performance")
    st.markdown(
        """
        | Category | Precision | Recall | F1 | Support |
        |----------|-----------|--------|----|---------|
        | defect | 0.92 | 0.88 | 0.90 | 417 |
        | shipping | 0.60 | 0.62 | 0.61 | 52 |
        | description | 0.57 | 0.65 | 0.61 | 51 |
        | size | 0.69 | 0.91 | 0.79 | 45 |
        | price | 0.70 | 0.85 | 0.77 | 92 |
        """
    )

    st.subheader("LLM Evaluation")
    eval_data = {
        "Metric": [
            "accuracy",
            "completeness",
            "actionability",
            "conciseness",
        ],
        "Base (Groq)": [3.61, 3.65, 4.22, 4.29],
        "Fine-tuned": [3.58, 3.58, 4.16, 4.28],
    }
    eval_df = pd.DataFrame(eval_data)
    st.bar_chart(
        eval_df.set_index("Metric"),
    )

    # Drift monitoring
    st.subheader("Data Drift")
    if os.path.exists("reports/drift_report.html"):
        st.markdown(
            "[Open Drift Report](reports/drift_report.html)"
        )
    else:
        st.info(
            "Run `poetry run python src/mlops/drift_monitor.py` "
            "to generate the drift report."
        )
