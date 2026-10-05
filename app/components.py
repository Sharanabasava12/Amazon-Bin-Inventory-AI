"""
Amazon Bin Inventory AI - UI Components for Streamlit Dashboard
Reusable widgets, KPI metric blocks, audit alert banners, and charting components.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def load_custom_css():
    """Loads custom CSS file for sleek logistics dark theme."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header():
    """Renders the top navigation brand header."""
    st.markdown(
        """
        <div class="header-box">
            <div>
                <h1 class="header-title">📦 Amazon Bin Inventory AI</h1>
                <div class="header-subtitle">
                    Automated Visual Item Counting & Inventory Discrepancy Auditing (aft-vbi-pds)
                </div>
            </div>
            <div>
                <span class="badge-live">● AI Vision Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audit_banner(audit_res: Dict[str, Any]):
    """Renders the primary visual audit verdict card."""
    severity = audit_res["severity"]
    title = audit_res["verdict_title"]
    message = audit_res["verdict_message"]
    action = audit_res["action_required"]

    css_class = f"audit-card-{severity}"

    st.markdown(
        f"""
        <div class="audit-card {css_class}">
            <div class="audit-title">{title}</div>
            <div class="audit-desc">{message}</div>
            <div style="margin-top: 8px; font-weight: 600; font-size: 13px;">
                ⚡ Recommendation: {action}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(audit_res: Dict[str, Any]):
    """Renders key metrics comparing Vision vs Warehouse Records."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Vision Item Count</div>
                <div class="kpi-value" style="color: #38bdf8;">{audit_res['predicted_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Recorded Quantity</div>
                <div class="kpi-value" style="color: #facc15;">{audit_res['expected_quantity']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        diff = audit_res["count_difference"]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        diff_color = "#4ade80" if diff == 0 else ("#f87171" if diff < 0 else "#fbbf24")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Discrepancy Delta</div>
                <div class="kpi-value" style="color: {diff_color};">{diff_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        conf_pct = audit_res["confidence"] * 100
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Model Confidence</div>
                <div class="kpi-value" style="color: #a78bfa;">{conf_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sku_table(skus: List[Dict]):
    """Displays structured warehouse catalog information for items in the bin."""
    if not skus:
        st.info("ℹ️ No catalog SKU items found in metadata for this bin.")
        return

    st.markdown("#### 🏷️ Warehouse Catalog Items in this Bin")
    for item in skus:
        name = item.get("name", "Unknown Item")
        asin = item.get("asin", "N/A")
        cat = item.get("product_type", "General")
        dim = item.get("dimensions", {})
        dim_str = f"{dim.get('length', '-')} × {dim.get('width', '-')} × {dim.get('height', '-')} cm" if dim else "-"

        st.markdown(
            f"""
            <div class="sku-item">
                <div>
                    <div class="sku-name">{name}</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                        Category: {cat} | Dimensions: {dim_str}
                    </div>
                </div>
                <div>
                    <span class="sku-asin">ASIN: {asin}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_chart(probabilities: Dict[str, float]):
    """Renders a clean probability bar chart showing model count confidence."""
    st.markdown("#### 📊 Prediction Probability Distribution")
    df = pd.DataFrame(
        list(probabilities.items()), columns=["Item Count Class", "Probability"]
    )
    df["Probability (%)"] = df["Probability"] * 100

    # Streamlit native bar chart
    st.bar_chart(df.set_index("Item Count Class")["Probability (%)"], color="#38bdf8")
