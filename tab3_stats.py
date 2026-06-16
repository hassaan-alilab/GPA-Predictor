# tab3_stats.py — Descriptive Statistics & Confidence Intervals

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["figure.facecolor"] = "#0f172a"
plt.rcParams["axes.facecolor"]   = "#1e293b"
plt.rcParams["axes.edgecolor"]   = "#334155"
plt.rcParams["axes.labelcolor"]  = "#94a3b8"
plt.rcParams["xtick.color"]      = "#64748b"
plt.rcParams["ytick.color"]      = "#64748b"
plt.rcParams["text.color"]       = "#cbd5e1"
plt.rcParams["axes.titlecolor"]  = "#93c5fd"
plt.rcParams["axes.titleweight"] = "bold"


def render_tab3(df, TARGET, FEATURES, ci_level):

    st.markdown('<p class="section-head">📐 Descriptive Statistics & Confidence Intervals</p>',
                unsafe_allow_html=True)

    # Only analyse numeric columns from FEATURES + TARGET
    num_cols = (
        df[FEATURES + [TARGET]]
        .select_dtypes(include=np.number)
        .columns.tolist()
    )

    # ── Descriptive statistics table ─────────────────────────
    st.markdown("#### Descriptive Statistical Measures")
    desc = df[num_cols].describe().T
    desc["median"]   = df[num_cols].median()
    desc["mode"]     = df[num_cols].mode().iloc[0]
    desc["variance"] = df[num_cols].var()
    desc["skewness"] = df[num_cols].skew()
    desc["kurtosis"] = df[num_cols].kurtosis()
    desc = desc[["count","mean","median","mode","std","variance","min","max","skewness","kurtosis"]]
    desc.columns = ["N","Mean","Median","Mode","Std Dev","Variance","Min","Max","Skewness","Kurtosis"]
    st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

    st.markdown(
        '<div class="highlight">'
        '💡 <b>Skewness</b> near 0 = symmetric. '
        '<b>Kurtosis</b> near 0 = normal tail weight. '
        '<b>Mode</b> = most frequent value.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Confidence intervals for every numeric column ────────
    st.markdown(f"#### Confidence Intervals — {int(ci_level * 100)}% Level")

    ci_rows = []
    for col in num_cols:
        data_col = df[col].dropna()
        n_col    = len(data_col)
        mean_col = data_col.mean()
        se_col   = stats.sem(data_col)
        lo, hi   = stats.t.interval(ci_level, df=n_col - 1, loc=mean_col, scale=se_col)
        ci_rows.append({
            "Variable":                          col,
            "N":                                 n_col,
            "Mean":                              round(mean_col, 4),
            "Std Error":                         round(se_col, 4),
            f"Lower ({int(ci_level*100)}% CI)":  round(lo, 4),
            f"Upper ({int(ci_level*100)}% CI)":  round(hi, 4),
            "Margin of Error":                   round(hi - mean_col, 4)
        })

    st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)

    st.divider()

    # ── CI error-bar chart for the target variable ───────────
    st.markdown(f"#### CI Visualisation — {TARGET}")

    fg             = df[TARGET].dropna()
    n_fg           = len(fg)
    mean_fg        = fg.mean()
    se_fg          = stats.sem(fg)
    lo_fg, hi_fg   = stats.t.interval(ci_level, df=n_fg - 1, loc=mean_fg, scale=se_fg)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.errorbar(
        mean_fg, 0,
        xerr=[[mean_fg - lo_fg], [hi_fg - mean_fg]],
        fmt="o", color="#60a5fa",
        capsize=12, capthick=2.5, elinewidth=2.5, markersize=10
    )
    ax.axvline(mean_fg, color="#f87171", linestyle="--", linewidth=1.5, label=f"Mean = {mean_fg:.4f}")
    ax.axvspan(lo_fg, hi_fg, alpha=0.08, color="#3b82f6")   # shaded confidence band
    ax.set_yticks([])
    ax.set_xlabel(TARGET)
    ax.set_title(f"{int(ci_level*100)}% CI for Mean {TARGET}  [{lo_fg:.4f} , {hi_fg:.4f}]")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f'<div class="highlight">'
        f'✅ We are <b>{int(ci_level*100)}%</b> confident that the true mean <b>{TARGET}</b> '
        f'lies between <b>{lo_fg:.4f}</b> and <b>{hi_fg:.4f}</b>.'
        f'</div>',
        unsafe_allow_html=True
    )