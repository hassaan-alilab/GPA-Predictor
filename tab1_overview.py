# tab1_overview.py — Dataset Overview: raw data, column info, frequency tables

import streamlit as st
import pandas as pd


def render_tab1(df, TARGET, FEATURES):

    st.markdown('<p class="section-head">📋 Dataset Overview</p>', unsafe_allow_html=True)

    # ── Summary metrics ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",  len(df))
    c2.metric("Total Variables", df.shape[1])
    c3.metric(f"Mean {TARGET}",  f"{df[TARGET].mean():.3f}")
    c4.metric("Std Dev of GPA",  f"{df[TARGET].std():.3f}")

    st.divider()

    # ── Raw data preview ─────────────────────────────────────
    st.markdown("#### 🗂️ Raw Data — First 20 Rows")
    st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    # ── Per-column diagnostics ───────────────────────────────
    st.markdown("#### 🔍 Column Information & Missing Values")
    info_df = pd.DataFrame({
        "Column":          df.columns,
        "Data Type":       df.dtypes.values,
        "Non-Null Count":  df.notnull().sum().values,
        "Missing":         df.isnull().sum().values,
        "Unique Values":   [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.divider()

    # ── Frequency table for Study Hours ─────────────────────
    if "StudyHoursPerWeek" in df.columns:
        st.markdown(f"#### 📊 Frequency Table — Study Hours Per Week")
        freq = df["StudyHoursPerWeek"].value_counts().sort_index().reset_index()
        freq.columns = ["StudyHoursPerWeek", "Count"]
        freq["Relative Frequency (%)"]   = (freq["Count"] / len(df) * 100).round(2)
        freq["Cumulative Frequency (%)"] = freq["Relative Frequency (%)"].cumsum().round(2)
        st.dataframe(freq, use_container_width=True)
        st.divider()

    # ── Summary of Health & Lifestyle Features ──────────────
    lifestyle_features = ["SocialMediaUsage", "PhysicalActivity", "SleepHours", "MentalHealthScore"]
    present_lifestyle = [f for f in lifestyle_features if f in df.columns]
    
    if present_lifestyle:
        st.markdown("#### 🏃 Health & Lifestyle — Descriptive Summary")
        lifestyle_summary = df[present_lifestyle].describe().T
        st.dataframe(lifestyle_summary.style.format("{:.3f}"), use_container_width=True)
        st.divider()

    # ── Download cleaned dataset ─────────────────────────────
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Processed Dataset",
        data=csv_bytes,
        file_name="processed_data.csv",
        mime="text/csv"
    )