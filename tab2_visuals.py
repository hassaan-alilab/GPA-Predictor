# tab2_visuals.py — All graphical representations (10 charts across 5 rows)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── Unified dark chart theme applied once for all figures ────
plt.rcParams["figure.facecolor"] = "#0f172a"
plt.rcParams["axes.facecolor"]   = "#1e293b"
plt.rcParams["axes.edgecolor"]   = "#334155"
plt.rcParams["axes.labelcolor"]  = "#94a3b8"
plt.rcParams["xtick.color"]      = "#64748b"
plt.rcParams["ytick.color"]      = "#64748b"
plt.rcParams["text.color"]       = "#cbd5e1"
plt.rcParams["axes.titlecolor"]  = "#93c5fd"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["grid.color"]       = "#334155"
plt.rcParams["grid.linestyle"]   = "--"
plt.rcParams["grid.alpha"]       = 0.4


def _fig(w=6, h=4):
    """Return a pre-styled (figure, axes) pair."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, axis="y", linewidth=0.5)
    return fig, ax


def render_tab2(df, TARGET, FEATURES):

    st.markdown('<p class="section-head">📈 Graphical Representations</p>', unsafe_allow_html=True)

    # ── Row 1: Histogram & Box Plot ──────────────────────────
    st.markdown("#### Distribution & Spread")
    col1, col2 = st.columns(2)

    with col1:
        st.caption(f"Histogram — {TARGET} Distribution")
        fig, ax = _fig()
        ax.hist(df[TARGET], bins=20, color="#3b82f6", edgecolor="#0f172a", alpha=0.85)
        ax.axvline(
            df[TARGET].mean(), color="#f87171", linestyle="--",
            linewidth=1.8, label=f"Mean = {df[TARGET].mean():.2f}"
        )
        ax.set_xlabel(TARGET)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {TARGET}")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        if "StudyHoursPerWeek" in df.columns:
            st.caption(f"Box Plot — {TARGET} by Study Hours (binned)")
            fig, ax = _fig()
            df_temp = df.copy()
            df_temp["StudyBin"] = pd.cut(df_temp["StudyHoursPerWeek"], bins=5)
            sns.boxplot(data=df_temp, x="StudyBin", y=TARGET, ax=ax, palette="Blues", linewidth=0.8)
            ax.set_xlabel("Study Hours (Binned)")
            ax.set_ylabel(TARGET)
            ax.set_title(f"{TARGET} by Study Hours")
            plt.xticks(rotation=28, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Row 2: Scatter Plots ─────────────────────────────────
    st.markdown("#### Study & Social Relationships")
    col3, col4 = st.columns(2)

    with col3:
        if "StudyHoursPerWeek" in df.columns:
            st.caption(f"Study Hours vs {TARGET}")
            fig, ax = _fig()
            ax.scatter(df["StudyHoursPerWeek"], df[TARGET], alpha=0.5, s=28, color="#60a5fa")
            ax.set_xlabel("Study Hours/Week")
            ax.set_ylabel(TARGET)
            ax.set_title(f"Study Hours vs {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col4:
        if "SocialMediaUsage" in df.columns:
            st.caption(f"Social Media Usage vs {TARGET}")
            fig, ax = _fig()
            ax.scatter(df["SocialMediaUsage"], df[TARGET], alpha=0.5, s=28, color="#fb923c")
            ax.set_xlabel("Social Media (hrs/day)")
            ax.set_ylabel(TARGET)
            ax.set_title(f"Social Media Usage vs {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Row 3: Heatmap & Attendance Scatter ──────
    st.markdown("#### Correlation & Attendance")
    col5, col6 = st.columns(2)

    with col5:
        st.caption("Correlation Heatmap — All Variables")
        num_df = df[FEATURES + [TARGET]].select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        sns.heatmap(
            num_df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.4, ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9, "color": "#f1f5f9"}
        )
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        if "AttendanceRate" in df.columns:
            st.caption(f"Attendance Rate vs {TARGET}")
            fig, ax = _fig()
            ax.scatter(df["AttendanceRate"], df[TARGET], color="#f43f5e", alpha=0.45, s=22)
            ax.set_xlabel("Attendance Rate (%)")
            ax.set_ylabel(TARGET)
            ax.set_title(f"Attendance Rate vs {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Row 4: Age & Physical Activity ──────
    st.markdown("#### Demographics & Lifestyle")
    col7, col8 = st.columns(2)

    with col7:
        if "Age" in df.columns:
            st.caption(f"Average {TARGET} by Age Group")
            avg_by_age = df.groupby("Age")[TARGET].mean()
            fig, ax = _fig()
            bars = ax.bar(avg_by_age.index, avg_by_age.values, color="#818cf8", edgecolor="#0f172a", width=0.65)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color="#cbd5e1")
            ax.set_xlabel("Age")
            ax.set_ylabel(f"Average {TARGET}")
            ax.set_title(f"Average {TARGET} by Age")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col8:
        if "PhysicalActivity" in df.columns:
            st.caption(f"Physical Activity vs {TARGET}")
            fig, ax = _fig()
            ax.scatter(df["PhysicalActivity"], df[TARGET], color="#10b981", alpha=0.45, s=22)
            ax.set_xlabel("Physical Activity (hrs/wk)")
            ax.set_ylabel(TARGET)
            ax.set_title(f"Physical Activity vs {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Row 5: Mental Health & Sleep ──────
    st.markdown("#### Health & Wellness")
    col9, col10 = st.columns(2)

    with col9:
        if "MentalHealthScore" in df.columns:
            st.caption(f"Mental Health vs Average {TARGET}")
            df_temp = df.copy()
            df_temp["MentalBin"] = pd.cut(df_temp["MentalHealthScore"], bins=5)
            line_data = (
                df_temp.groupby("MentalBin", observed=True)[TARGET]
                .mean()
                .reset_index()
            )
            fig, ax = _fig()
            ax.plot(
                range(len(line_data)), line_data[TARGET],
                color="#f472b6", linewidth=2.2,
                marker="o", markersize=6, markerfacecolor="#db2777"
            )
            ax.fill_between(range(len(line_data)), line_data[TARGET], alpha=0.15, color="#f472b6")
            ax.set_xticks(range(len(line_data)))
            ax.set_xticklabels(
                [str(b) for b in line_data["MentalBin"]],
                rotation=30, ha="right", fontsize=7.5
            )
            ax.set_xlabel("Mental Health Score")
            ax.set_ylabel(f"Average {TARGET}")
            ax.set_title(f"Mental Health vs Average {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col10:
        if "SleepHours" in df.columns:
            st.caption(f"Sleep Hours vs {TARGET}")
            fig, ax = _fig()
            ax.scatter(df["SleepHours"], df[TARGET], color="#8b5cf6", alpha=0.45, s=22)
            ax.set_xlabel("Sleep Hours (night)")
            ax.set_ylabel(TARGET)
            ax.set_title(f"Sleep Hours vs {TARGET}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)