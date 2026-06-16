# tab4_distribution.py — Probability distributions, normality checks, CDF calculator

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

plt.rcParams["figure.facecolor"] = "#0f172a"
plt.rcParams["axes.facecolor"]   = "#1e293b"
plt.rcParams["axes.edgecolor"]   = "#334155"
plt.rcParams["axes.labelcolor"]  = "#94a3b8"
plt.rcParams["xtick.color"]      = "#64748b"
plt.rcParams["ytick.color"]      = "#64748b"
plt.rcParams["text.color"]       = "#cbd5e1"
plt.rcParams["axes.titlecolor"]  = "#93c5fd"
plt.rcParams["axes.titleweight"] = "bold"


def render_tab4(df, TARGET):

    st.markdown('<p class="section-head">🔔 Probability Methods & Distributions</p>',
                unsafe_allow_html=True)

    # Pre-compute distribution parameters for the target variable
    fg    = df[TARGET].dropna()
    mu    = fg.mean()
    sigma = fg.std()
    x     = np.linspace(fg.min() - 0.5, fg.max() + 0.5, 300)

    # ── Normal fit & Q-Q Plot ────────────────────────────────
    st.markdown("#### Normal Distribution Fit & Normality Check")
    col1, col2 = st.columns(2)

    with col1:
        st.caption(f"Normal Fit — {TARGET}")
        pdf = norm.pdf(x, mu, sigma)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.hist(fg, bins=20, density=True, color="#3b82f6", edgecolor="#0f172a", alpha=0.75, label="Observed")
        ax.plot(x, pdf, color="#f87171", linewidth=2.2, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
        ax.set_xlabel(TARGET)
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Normal Distribution Fit — {TARGET}")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            f'<div class="highlight">'
            f'μ = <b>{mu:.4f}</b> &nbsp;|&nbsp; σ = <b>{sigma:.4f}</b> '
            f'&nbsp;|&nbsp; Skewness = <b>{fg.skew():.4f}</b> '
            f'&nbsp;|&nbsp; Kurtosis = <b>{fg.kurtosis():.4f}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.caption(f"Q-Q Plot — Normality Check ({TARGET})")
        (osm, osr), (slope, intercept, r) = stats.probplot(fg, dist="norm")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.plot(osm, osr, "o", color="#60a5fa", markersize=4.5, alpha=0.65, label="Data quantiles")
        ax.plot(osm, slope * np.array(osm) + intercept, color="#f87171", linewidth=2, label="Normal reference")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"Q-Q Plot of {TARGET}")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Shapiro-Wilk test — limited to 200 samples as required by the test
        sample         = fg.sample(min(200, len(fg)), random_state=42)
        stat_sw, p_sw  = stats.shapiro(sample)
        normal_str     = "✅ likely normal (p > 0.05)" if p_sw > 0.05 else "⚠️ not perfectly normal (p ≤ 0.05)"
        st.markdown(
            f'<div class="highlight">'
            f'Shapiro-Wilk: W = <b>{stat_sw:.4f}</b>, p = <b>{p_sw:.4f}</b> → {normal_str}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Interactive probability calculator ───────────────────
    st.markdown("#### 🎯 Probability Calculator")
    c1, c2 = st.columns(2)

    with c1:
        grade_threshold = st.slider(
            f"Set {TARGET} threshold",
            float(df[TARGET].min()), float(df[TARGET].max()),
            float(mu), step=0.05
        )

    with c2:
        p_below = norm.cdf(grade_threshold, mu, sigma)
        p_above = 1 - p_below
        st.metric(f"P({TARGET} ≤ {grade_threshold:.2f})", f"{p_below:.4f}  ({p_below*100:.1f}%)")
        st.metric(f"P({TARGET} > {grade_threshold:.2f})", f"{p_above:.4f}  ({p_above*100:.1f}%)")

    # ── CDF chart with shaded region ─────────────────────────
    st.markdown(f"**Cumulative Distribution Function (CDF) — {TARGET}**")
    cdf = norm.cdf(x, mu, sigma)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.plot(x, cdf, color="#60a5fa", linewidth=2.5)
    ax.fill_between(x, cdf, where=(x <= grade_threshold), alpha=0.18, color="#3b82f6")
    ax.axvline(grade_threshold, color="#f87171", linestyle="--", linewidth=1.8, label=f"Threshold = {grade_threshold:.2f}")
    ax.axhline(p_below, color="#fbbf24", linestyle=":", linewidth=1.5, label=f"P = {p_below:.4f}")
    ax.set_xlabel(TARGET)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"CDF of {TARGET}")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Social Media Usage distribution ─────────────────────
    if "SocialMediaUsage" in df.columns:
        st.divider()
        st.markdown("#### 📱 Social Media Usage — Probability Distribution")
        sm    = df["SocialMediaUsage"].dropna()
        mu_s  = sm.mean()
        sig_s = sm.std()
        xs    = np.linspace(sm.min() - 0.5, sm.max() + 0.5, 300)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.hist(sm, bins=20, density=True, color="#a78bfa", edgecolor="#0f172a", alpha=0.7, label="Observed")
        ax.plot(xs, norm.pdf(xs, mu_s, sig_s), color="#f87171", linewidth=2.2, label=f"N(μ={mu_s:.2f}, σ={sig_s:.2f})")
        ax.set_xlabel("Social Media Usage (hrs/day)")
        ax.set_ylabel("Density")
        ax.set_title("Normal Fit — Social Media Usage")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)