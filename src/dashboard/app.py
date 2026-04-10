"""Product Intelligence Dashboard - Production UI.

5 pages: Quality Alerts, Product Deep Dive,
Classifier Demo, Semantic Search, Model Performance.
"""

import os
import re
import sys
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Add project root to path for imports
_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="Product Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database ────────────────────────────────────────


def _get_secret(key, default=None):
    """Read from Streamlit secrets or env vars."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


@st.cache_resource
def get_engine():
    neon_url = _get_secret("NEON_DATABASE_URL")
    if neon_url:
        return create_engine(neon_url)
    host = _get_secret("POSTGRES_HOST", "localhost")
    port = _get_secret("POSTGRES_PORT", "5432")
    db = _get_secret("POSTGRES_DB", "product_intelligence")
    user = _get_secret("POSTGRES_USER", "postgres")
    pw = _get_secret("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def _db_ok() -> tuple:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "PostgreSQL connected"
    except Exception:
        return False, "PostgreSQL unavailable"


# ── Render helpers (matching reference style) ───────


def render_page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header-block">'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(cards):
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        vc = ""
        if c.get("color"):
            vc = f' style="color:{c["color"]}"'
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">'
                f'{c["label"]}</div>'
                f'<div class="metric-value"{vc}>'
                f'{c["value"]}</div>'
                f'<div class="metric-delta">'
                f'{c.get("delta", "")}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_section_divider(label):
    st.markdown(
        f'<div class="section-divider">'
        f'<span class="section-divider-label">'
        f"{label}</span>"
        f'<div class="section-divider-line"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(
        f'<div class="table-wrap"><table>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


def badge(text_val, cls="badge-green"):
    return f'<span class="badge {cls}">{text_val}</span>'


def mono(text_val):
    return f'<span class="text-mono">{text_val}</span>'


# ── Cached queries ──────────────────────────────────


@st.cache_data(ttl=300)
def load_alerts(severity=None):
    q = "SELECT * FROM alerts ORDER BY detected_at DESC LIMIT 200"
    if severity and severity != "All":
        q = (
            "SELECT * FROM alerts "
            "WHERE severity = :sev "
            "ORDER BY detected_at DESC LIMIT 200"
        )
        return pd.read_sql(text(q), get_engine(), params={"sev": severity})
    return pd.read_sql(text(q), get_engine())


@st.cache_data(ttl=300)
def load_top_products(limit=50):
    q = text(
        """
        SELECT r.asin, p.title,
               COUNT(*) as review_count,
               ROUND(AVG(r.rating)::numeric, 2) as avg_rating
        FROM reviews r
        LEFT JOIN products p ON r.parent_asin = p.parent_asin
        GROUP BY r.asin, p.title
        ORDER BY review_count DESC
        LIMIT :lim
        """
    )
    return pd.read_sql(q, get_engine(), params={"lim": limit})


@st.cache_data(ttl=300)
def load_reviews_for_product(asin, limit=20):
    q = text(
        "SELECT rating, title, text, helpful_votes, timestamp "
        "FROM reviews WHERE asin = :asin "
        "ORDER BY helpful_votes DESC LIMIT :lim"
    )
    return pd.read_sql(q, get_engine(), params={"asin": asin, "lim": limit})


@st.cache_data(ttl=300)
def load_rating_distribution(asin):
    q = text(
        "SELECT rating, COUNT(*) as count "
        "FROM reviews WHERE asin = :asin "
        "GROUP BY rating ORDER BY rating"
    )
    return pd.read_sql(q, get_engine(), params={"asin": asin})


# ── Navigation ──────────────────────────────────────

_NAV = [
    "◈ Quality Alerts",
    "◆ Product Deep Dive",
    "✦ Classifier Demo",
    "◎ Semantic Search",
    "▣ Model Performance",
]

# ── Sidebar ─────────────────────────────────────────

with st.sidebar:
    ok, status_msg = _db_ok()
    dot = "status-green" if ok else "status-red"
    st.markdown(
        f'<div style="margin-top:-20px;padding-top:0">'
        f'<div class="sidebar-brand">Product Intelligence</div>'
        f'<div class="sidebar-title">Review Analytics Platform</div>'
        f'<div class="sidebar-status-row">'
        f'<span class="status-dot {dot}"></span>'
        f"{status_msg}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### Navigate")
    page = st.radio(
        "Nav",
        _NAV,
        label_visibility="collapsed",
        key="page_radio",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#5a5955;'
        'line-height:1.6">'
        "Multi-Agent Product Analytics<br>"
        "500K reviews · 1.6M products<br>"
        "PyTorch · Mistral-7B · LangGraph<br>"
        "FastAPI · Streamlit</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════
# PAGE 1 — Quality Alerts
# ═══════════════════════════════════════════════════


def page_alerts():
    render_page_header(
        "Quality Alerts",
        "Products flagged by the anomaly detection autoencoder",
    )

    severity = st.selectbox(
        "Filter by severity",
        ["All", "critical", "warning"],
    )

    with st.spinner("Loading alerts..."):
        try:
            alerts = load_alerts(severity)
        except Exception as e:
            st.error(f"Could not load alerts: {e}")
            return

    if alerts.empty:
        st.info("No alerts found.")
        return

    total = len(alerts)
    critical = len(alerts[alerts["severity"] == "critical"])
    warning = len(alerts[alerts["severity"] == "warning"])

    render_metric_cards(
        [
            {
                "label": "Total Alerts",
                "value": f"{total:,}",
                "delta": "From anomaly detector",
            },
            {
                "label": "Critical",
                "value": f"{critical:,}",
                "delta": "Score > 2x threshold",
                "color": "#dc2626",
            },
            {
                "label": "Warning",
                "value": f"{warning:,}",
                "delta": "Score > threshold",
                "color": "#d97706",
            },
        ]
    )

    render_section_divider("ALERT DETAILS")

    tbl_rows = []
    for _, r in alerts.head(50).iterrows():
        sev = r.get("severity", "")
        sev_badge = (
            badge(sev, "badge-red") if sev == "critical" else badge(sev, "badge-amber")
        )
        tbl_rows.append(
            [
                mono(r.get("product_id", "")),
                r.get("alert_type", ""),
                sev_badge,
                str(r.get("detected_at", ""))[:19],
                str(r.get("details", ""))[:80],
            ]
        )

    render_table(
        ["Product", "Type", "Severity", "Detected", "Details"],
        tbl_rows,
    )


# ═══════════════════════════════════════════════════
# PAGE 2 — Product Deep Dive
# ═══════════════════════════════════════════════════


def page_deep_dive():
    render_page_header(
        "Product Deep Dive",
        "Select a product to analyze reviews, NER entities, and complaint patterns",
    )

    with st.spinner("Loading products..."):
        top_products = load_top_products()

    if top_products.empty:
        st.info("No products found.")
        return

    selected = st.selectbox(
        "Select product",
        top_products["asin"].tolist(),
        format_func=lambda x: (
            f"{x} — "
            + str(
                top_products[top_products["asin"] == x]["title"].values[0] or "Unknown"
            )[:60]
        ),
    )

    if not selected:
        return

    info = top_products[top_products["asin"] == selected].iloc[0]

    avg_r = float(info["avg_rating"])
    rating_color = (
        "#0d9f6e" if avg_r >= 4.0 else "#d97706" if avg_r >= 3.0 else "#dc2626"
    )

    render_metric_cards(
        [
            {
                "label": "Total Reviews",
                "value": f"{info['review_count']:,}",
                "delta": "",
            },
            {
                "label": "Avg Rating",
                "value": f"{avg_r}",
                "delta": "out of 5.0",
                "color": rating_color,
            },
            {
                "label": "Product",
                "value": (
                    '<span style="font-size:14px">'
                    + str(info["title"] or "Unknown")[:60]
                    + "</span>"
                ),
                "delta": selected,
            },
        ]
    )

    # Rating distribution chart
    render_section_divider("RATING DISTRIBUTION")
    with st.spinner("Loading distribution..."):
        dist = load_rating_distribution(selected)

    if not dist.empty:
        colors = {
            1: "#dc2626",
            2: "#f97316",
            3: "#d97706",
            4: "#0d9488",
            5: "#0d9f6e",
        }
        fig = go.Figure()
        for _, row in dist.iterrows():
            r_val = int(row["rating"])
            fig.add_trace(
                go.Bar(
                    x=[f"{r_val} star"],
                    y=[row["count"]],
                    marker_color=colors.get(r_val, "#6366f1"),
                    name=f"{r_val} star",
                    showlegend=False,
                )
            )
        fig.update_layout(
            height=250,
            margin=dict(t=10, b=30, l=40, r=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    # NER Analysis
    render_section_divider("NER ENTITY ANALYSIS")
    with st.spinner("Running NER..."):
        try:
            reviews = load_reviews_for_product(selected)
            try:
                from src.features.ner_extractor import (
                    extract_batch_fast,
                )
            except ImportError:
                _dash_dir = os.path.dirname(os.path.abspath(__file__))
                if _dash_dir not in sys.path:
                    sys.path.insert(0, _dash_dir)
                from ner_inline import extract_batch_fast

            texts = reviews["text"].fillna("").tolist()
            ner_results = extract_batch_fast(texts)

            comp_counter = Counter()
            issue_counter = Counter()
            for ner in ner_results:
                for c in ner["components"]:
                    comp_counter[c] += 1
                for i in ner["issues"]:
                    issue_counter[i] += 1

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    '<div class="card-title">Top Components</div>',
                    unsafe_allow_html=True,
                )
                comp_html = " ".join(
                    badge(f"{c} ({n})", "badge-blue")
                    for c, n in comp_counter.most_common(10)
                )
                st.markdown(
                    f'<div class="card">{comp_html}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    '<div class="card-title">Top Issues</div>',
                    unsafe_allow_html=True,
                )
                issue_html = " ".join(
                    badge(f"{i} ({n})", "badge-red")
                    for i, n in issue_counter.most_common(10)
                )
                st.markdown(
                    f'<div class="card">{issue_html}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"NER analysis unavailable: {e}")

    # Top Reviews
    render_section_divider("TOP REVIEWS")
    if not reviews.empty:
        for _, row in reviews.head(10).iterrows():
            r_val = int(row["rating"])
            star_cls = (
                "badge-green"
                if r_val >= 4
                else "badge-amber" if r_val >= 3 else "badge-red"
            )
            stars = "★" * r_val + "☆" * (5 - r_val)
            title = row["title"] or "No title"
            review_text = str(row["text"] or "")[:300]
            helpful = row.get("helpful_votes", 0)

            st.markdown(
                f'<div class="card">'
                f'<div class="flex items-center gap-8 mb-4">'
                f"{badge(stars, star_cls)} "
                f'<span style="font-weight:600">{title}</span>'
                f"</div>"
                f'<div class="text-sm" style="color:#4b4a45">'
                f"{review_text}...</div>"
                f'<div class="text-xs text-muted" style="margin-top:8px">'
                f"Helpful votes: {helpful}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════
# PAGE 3 — Classifier Demo
# ═══════════════════════════════════════════════════


def page_classifier():
    render_page_header(
        "Root Cause Classifier",
        "Paste any review text to classify its root cause into 5 categories",
    )

    examples = {
        "Defect": (
            "The battery died after 2 weeks and the "
            "screen started flickering. Completely broken."
        ),
        "Shipping": (
            "Package arrived damaged with missing parts. "
            "Box was crushed and the item was broken."
        ),
        "Description": (
            "This looks nothing like the picture. "
            "The description said waterproof but it "
            "doesn't match at all. Misleading listing."
        ),
        "Size": (
            "Way too small for what I expected. " "Doesn't fit at all. Wrong size."
        ),
        "Price": (
            "Total waste of money. Overpriced garbage. "
            "You can find better for half the price. Scam."
        ),
        "Multiple Issues": (
            "The battery died after 2 weeks, screen is "
            "cracked, arrived damaged with missing parts. "
            "Total waste of money and doesn't match "
            "the description at all."
        ),
    }

    render_section_divider("EXAMPLES")
    ex_cols = st.columns(len(examples))
    for col, (name, txt) in zip(ex_cols, examples.items()):
        with col:
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                st.session_state["demo_text"] = txt

    review_text = st.text_area(
        "Enter review text",
        value=st.session_state.get("demo_text", ""),
        height=120,
    )

    if st.button("Classify", type="primary") and review_text:
        with st.spinner("Classifying..."):
            try:
                try:
                    from src.features.ner_extractor import (
                        extract_fast,
                    )
                except ImportError:
                    _dash_dir = os.path.dirname(os.path.abspath(__file__))
                    if _dash_dir not in sys.path:
                        sys.path.insert(0, _dash_dir)
                    from ner_inline import extract_fast

                ner = extract_fast(review_text)

                render_section_divider("NER ENTITIES")
                col1, col2 = st.columns(2)
                with col1:
                    comps = (
                        " ".join(badge(c, "badge-blue") for c in ner["components"])
                        or '<span class="text-muted">None detected</span>'
                    )
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="metric-label">Components</div>'
                        f"{comps}</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    issues = (
                        " ".join(badge(i, "badge-red") for i in ner["issues"])
                        or '<span class="text-muted">None detected</span>'
                    )
                    st.markdown(
                        f'<div class="card">'
                        f'<div class="metric-label">Issues</div>'
                        f"{issues}</div>",
                        unsafe_allow_html=True,
                    )

                render_section_divider("ROOT CAUSE CLASSIFICATION")

                cat_config = {
                    "defect": {
                        "color": "#dc2626",
                        "bg": "#fee2e2",
                        "words": [
                            "broke",
                            "broken",
                            "cracked",
                            "defective",
                            "faulty",
                            "malfunction",
                            "dead",
                            "died",
                            "stopped working",
                            "not working",
                            "doesn't work",
                            "does not work",
                            "won't turn on",
                            "won't charge",
                            "overheating",
                            "overheat",
                            "slow",
                            "laggy",
                            "freezing",
                            "crash",
                            "crashed",
                            "glitchy",
                            "buggy",
                            "discharged",
                            "drains",
                            "unresponsive",
                        ],
                    },
                    "shipping": {
                        "color": "#2563eb",
                        "bg": "#dbeafe",
                        "words": [
                            "arrived damaged",
                            "arrived broken",
                            "missing parts",
                            "wrong item",
                            "wrong product",
                            "late delivery",
                            "never arrived",
                            "incomplete",
                            "damaged in transit",
                        ],
                    },
                    "description": {
                        "color": "#7c3aed",
                        "bg": "#ede9fe",
                        "words": [
                            "misleading",
                            "not as described",
                            "not as advertised",
                            "false advertising",
                            "cheaply made",
                            "cheap quality",
                            "poor quality",
                            "low quality",
                            "flimsy",
                            "looks nothing like",
                            "doesn't match",
                            "does not match",
                            "description",
                            "listing",
                        ],
                    },
                    "size": {
                        "color": "#d97706",
                        "bg": "#fef3c7",
                        "words": [
                            "too small",
                            "too big",
                            "too large",
                            "too tight",
                            "doesn't fit",
                            "does not fit",
                            "wrong size",
                        ],
                    },
                    "price": {
                        "color": "#0d9488",
                        "bg": "#cffafe",
                        "words": [
                            "waste of money",
                            "overpriced",
                            "rip off",
                            "ripoff",
                            "scam",
                            "not worth",
                            "worthless",
                        ],
                    },
                }

                text_lower = review_text.lower()
                ner_issues = set(ner.get("issues", []))

                for cat, cfg in cat_config.items():
                    score = 0.0
                    for w in cfg["words"]:
                        if w in text_lower or w in ner_issues:
                            score += 0.2
                    score = min(1.0, score)
                    pct = int(score * 100)

                    st.markdown(
                        f'<div style="margin-bottom:10px">'
                        f'<div class="flex items-center gap-8">'
                        f'<span style="width:100px;font-weight:600;'
                        f'font-size:14px;text-transform:capitalize">'
                        f"{cat}</span>"
                        f'<div style="flex:1;background:#f0f0ec;'
                        f'border-radius:8px;height:28px;overflow:hidden">'
                        f'<div style="width:{pct}%;height:100%;'
                        f"background:{cfg['color']};border-radius:8px;"
                        f'transition:width 0.5s ease"></div></div>'
                        f'<span style="width:50px;text-align:right;'
                        f"font-weight:700;font-size:14px;"
                        f"color:{cfg['color']}\">{pct}%</span>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Classification failed: {e}")


# ═══════════════════════════════════════════════════
# PAGE 4 — Semantic Search
# ═══════════════════════════════════════════════════


def page_search():
    render_page_header(
        "Semantic Search",
        "Search 50K review embeddings by meaning, not just keywords",
    )

    # Check if search dependencies are available
    try:
        import torch  # noqa: F401
    except ImportError:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">Semantic Search</div>'
            '<div class="text-sm" style="color:#4b4a45">'
            "This feature requires PyTorch and "
            "sentence-transformers which are available "
            "in the local deployment.<br><br>"
            "<b>How it works:</b> Natural language queries "
            "like 'bluetooth keeps disconnecting' are "
            "encoded into 384-dim vectors and matched "
            "against 8,963 review embeddings in Pinecone "
            "using cosine similarity.<br><br>"
            "<b>To try locally:</b><br>"
            "<code>docker-compose up -d</code><br>"
            "<code>poetry run streamlit run "
            "src/dashboard/app.py</code>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return


    search_examples = {
        "Bluetooth": "bluetooth keeps disconnecting",
        "Battery": "battery drains too fast",
        "Screen": "screen flickering and dim",
        "Charging": "won't charge after a month",
        "Sound": "terrible sound quality",
        "Size": "too small doesn't fit",
    }

    render_section_divider("TRY THESE")
    ex_cols = st.columns(len(search_examples))
    for col, (name, txt) in zip(ex_cols, search_examples.items()):
        with col:
            if st.button(
                name,
                key=f"se_{name}",
                use_container_width=True,
            ):
                st.session_state["search_q"] = txt

    query = st.text_input(
        "Search query",
        value=st.session_state.get("search_q", ""),
        placeholder="e.g., bluetooth keeps disconnecting",
    )
    max_rating = st.slider("Max rating filter", 1.0, 5.0, 3.0, 0.5)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching embeddings..."):
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
                    return

                st.markdown(
                    f'<div class="text-sm text-muted" '
                    f'style="margin-bottom:14px">'
                    f"Found {len(results)} results</div>",
                    unsafe_allow_html=True,
                )

                for r in results:
                    sim = 1 - r["distance"]
                    sim_cls = (
                        "badge-green"
                        if sim > 0.7
                        else "badge-amber" if sim > 0.5 else "badge-red"
                    )
                    r_val = int(r["rating"])
                    star_cls = (
                        "badge-green"
                        if r_val >= 4
                        else "badge-amber" if r_val >= 3 else "badge-red"
                    )
                    stars = "★" * r_val + "☆" * (5 - r_val)

                    st.markdown(
                        f'<div class="card">'
                        f'<div class="flex items-center gap-8 mb-4">'
                        f"{badge(stars, star_cls)} "
                        f'{badge(f"Similarity: {sim:.2f}", sim_cls)} '
                        f'{mono(r["asin"])}'
                        f"</div>"
                        f'<div style="font-weight:600;margin-bottom:6px">'
                        f'{r.get("title", "")}</div>'
                        f'<div class="text-sm" style="color:#4b4a45">'
                        f'{r["text"][:300]}...</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Search failed: {e}")


# ═══════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ═══════════════════════════════════════════════════


def page_performance():
    render_page_header(
        "Model Performance",
        "Metrics, evaluation results, and drift monitoring",
    )

    render_section_divider("MODEL METRICS")

    render_metric_cards(
        [
            {
                "label": "Root Cause Classifier",
                "value": "F1 = 0.7339",
                "delta": badge("PASS", "badge-green"),
                "color": "#0d9f6e",
            },
            {
                "label": "Anomaly Detector",
                "value": "21,310 alerts",
                "delta": badge("PASS", "badge-green"),
                "color": "#0d9f6e",
            },
            {
                "label": "Helpfulness Predictor",
                "value": "MAE = 1.46",
                "delta": badge("PASS", "badge-green"),
                "color": "#0d9f6e",
            },
            {
                "label": "Fine-tuned Mistral",
                "value": "3.90 / 5",
                "delta": badge("PASS", "badge-green"),
                "color": "#0d9f6e",
            },
        ]
    )

    render_section_divider("CLASSIFIER PER-CATEGORY")

    render_table(
        [
            "Category",
            "Precision",
            "Recall",
            "F1-Score",
            "Support",
        ],
        [
            [
                badge("defect", "badge-red"),
                "0.92",
                "0.88",
                '<span style="font-weight:700">0.90</span>',
                "417",
            ],
            [
                badge("shipping", "badge-blue"),
                "0.60",
                "0.62",
                '<span style="font-weight:700">0.61</span>',
                "52",
            ],
            [
                badge("description", "badge-purple"),
                "0.57",
                "0.65",
                '<span style="font-weight:700">0.61</span>',
                "51",
            ],
            [
                badge("size", "badge-amber"),
                "0.69",
                "0.91",
                '<span style="font-weight:700">0.79</span>',
                "45",
            ],
            [
                badge("price", "badge-teal"),
                "0.70",
                "0.85",
                '<span style="font-weight:700">0.77</span>',
                "92",
            ],
        ],
    )

    render_section_divider("LLM EVALUATION")

    eval_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Completeness",
                "Actionability",
                "Conciseness",
            ],
            "Base (Groq)": [3.61, 3.65, 4.22, 4.29],
            "Fine-tuned": [3.58, 3.58, 4.16, 4.28],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Base (Groq)",
            x=eval_data["Metric"],
            y=eval_data["Base (Groq)"],
            marker_color="#6366f1",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fine-tuned",
            x=eval_data["Metric"],
            y=eval_data["Fine-tuned"],
            marker_color="#14b8a6",
        )
    )
    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(t=30, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif"),
        yaxis=dict(range=[0, 5], title="Score (1-5)"),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drift monitoring
    render_section_divider("DATA DRIFT MONITORING")
    report_path = os.path.join(_ROOT, "reports", "drift_report.html")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        scaled_html = (
            '<div style="transform:scale(0.65);'
            "transform-origin:top left;"
            'width:154%;height:154%">'
            f"{html_content}</div>"
        )
        st.components.v1.html(scaled_html, height=900, scrolling=True)
    else:
        st.info(
            "Drift report not found. "
            "Run `poetry run python src/mlops/drift_monitor.py` "
            "to generate it."
        )


# ═══════════════════════════════════════════════════
# CSS — matching reference style
# ═══════════════════════════════════════════════════

st.markdown(
    """<style>
:root{--green:#0d9f6e;--red:#dc2626;--amber:#d97706;
--blue:#2563eb;--teal:#0d9488;--purple:#7c3aed}
.stApp{background:#f8f7f4;color:#1a1a1e}
.block-container{max-width:1100px;padding-top:50px;
padding-bottom:50px}
footer{visibility:hidden}
[data-testid="stSidebar"]>div:first-child{
padding:0px 14px 14px !important;background:#1a1a1e}
[data-testid="stSidebar"] *{color:#e0dfd8}
[data-testid="stSidebar"] label{font-size:11px !important;
text-transform:uppercase;letter-spacing:.8px;
color:#7a7972 !important}
[data-testid="stSidebar"] button{color:#1a1a1e !important;
background:#d4d3ce !important;border:1px solid #9c9a92 !important;
font-weight:600 !important}
[data-testid="stSidebar"] button:hover{
background:#ffffff !important;color:#1a1a1e !important}
[data-testid="stSidebar"] hr{margin:6px 0 !important}
[data-testid="stSidebar"] [role="radiogroup"] label{
background:transparent !important;border:none !important;
padding:6px 0 !important;margin:0 !important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]>div{
background:#2a2a2f !important;
border:1px solid rgba(255,255,255,0.06) !important;
border-radius:6px !important;color:#e0dfd8 !important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span{
color:#e0dfd8 !important}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg{
fill:#e0dfd8 !important}
.stButton>button[kind="primary"]{background-color:#6366f1 !important;
border-color:#6366f1 !important;color:#fff !important}
.stButton>button[kind="primary"]:hover{
background-color:#4f46e5 !important}
.sidebar-brand{font-size:13px;font-weight:500;
letter-spacing:2px;text-transform:uppercase;
color:#9c9a92;margin-bottom:4px}
.sidebar-title{font-size:20px;font-weight:600;
color:#fff;margin-bottom:8px}
.sidebar-status-row{font-size:12px;color:#d1d5db;
margin-bottom:14px;display:flex;align-items:center;gap:8px}
.status-dot{display:inline-block;width:7px;height:7px;
border-radius:50%}
.status-green{background:#10b981}
.status-red{background:#ef4444}
.page-header-block{margin-bottom:10px}
.page-title{font-size:24px;font-weight:600;color:#1a1a1e}
.page-subtitle{font-size:13px;color:#9c9a92;
margin-top:2px;margin-bottom:16px}
.metric-card{background:#fff;border-radius:10px;
padding:16px 18px;border:1px solid rgba(0,0,0,0.08);
margin-bottom:14px;min-height:110px}
.metric-card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.06)}
.metric-label{font-size:12px;text-transform:uppercase;
letter-spacing:.6px;color:#6b6a65;margin-bottom:4px;
font-weight:500}
.metric-value{font-size:24px;font-weight:700;line-height:1.2;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric-delta{font-size:13px;margin-top:3px}
.text-muted{color:#9c9a92}
.text-mono{font-family:Consolas,monospace}
.text-xs{font-size:12px}.text-sm{font-size:13px}
.mb-4{margin-bottom:4px}.mb-8{margin-bottom:8px}
.flex{display:flex}.items-center{align-items:center}
.gap-8{gap:8px}
.badge{display:inline-flex;align-items:center;
padding:4px 12px;border-radius:20px;font-size:13px;
font-weight:700;letter-spacing:0.2px;margin:2px}
.badge-green{background:#d1fae5;color:#065f46}
.badge-amber{background:#fef3c7;color:#7c2d12}
.badge-red{background:#fee2e2;color:#991b1b}
.badge-blue{background:#dbeafe;color:#1e40af}
.badge-purple{background:#ede9fe;color:#6d28d9}
.badge-teal{background:#cffafe;color:#155e75}
.card{background:#fff;border:1px solid rgba(0,0,0,0.08);
border-radius:14px;padding:20px 22px;margin-bottom:16px}
.card-title{font-size:16px;font-weight:600;
margin-bottom:14px;color:#1a1a1e}
.table-wrap{overflow-x:auto;border-radius:10px;
border:1px solid #e8e7e3;background:#fff}
table{width:100%;border-collapse:collapse;font-size:14px}
thead th{text-align:left;padding:12px 14px;font-size:12px;
text-transform:uppercase;letter-spacing:.5px;color:#6b6a65;
font-weight:600;background:#fafaf8;
border-bottom:1px solid #e8e7e3}
tbody td{padding:12px 14px;border-bottom:1px solid #e8e7e3;
color:#1a1a2e;font-size:14px}
tbody tr:last-child td{border-bottom:none}
.section-divider{display:flex;align-items:center;
gap:12px;margin:36px 0 20px}
.section-divider-line{flex:1;height:1px;
background:rgba(0,0,0,0.08)}
.section-divider-label{font-size:12px;
text-transform:uppercase;letter-spacing:1.5px;
color:#6b6a65;white-space:nowrap;font-weight:600}
.stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"]>div{
color:#1a1a1e !important;background:#fff !important;
border:1px solid #d4d3ce !important}
.stTextArea textarea::placeholder,.stTextInput input::placeholder{
color:#9c9a92 !important}
</style>""",
    unsafe_allow_html=True,
)

# ── Page dispatch ───────────────────────────────────

_DISPATCH = {
    "◈ Quality Alerts": page_alerts,
    "◆ Product Deep Dive": page_deep_dive,
    "✦ Classifier Demo": page_classifier,
    "◎ Semantic Search": page_search,
    "▣ Model Performance": page_performance,
}

_DISPATCH.get(page, page_alerts)()
