# tab5_model.py — Linear Regression: training, evaluation, diagnostics, live prediction

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams["figure.facecolor"] = "#0f172a"
plt.rcParams["axes.facecolor"]   = "#1e293b"
plt.rcParams["axes.edgecolor"]   = "#334155"
plt.rcParams["axes.labelcolor"]  = "#94a3b8"
plt.rcParams["xtick.color"]      = "#64748b"
plt.rcParams["ytick.color"]      = "#64748b"
plt.rcParams["text.color"]       = "#cbd5e1"
plt.rcParams["axes.titlecolor"]  = "#93c5fd"
plt.rcParams["axes.titleweight"] = "bold"


def render_tab5(df, TARGET, FEATURES, test_size, random_seed):

    st.markdown('<p class="section-head">🤖 Regression Modeling & Predictions</p>',
                unsafe_allow_html=True)

    # ── Prepare feature matrix and target vector ─────────────
    X     = df[FEATURES].copy()
    y     = df[TARGET].copy()
    valid = X.notnull().all(axis=1) & y.notnull()   # drop rows with any missing value
    X     = X[valid]
    y     = y[valid]

    # ── 80 / 20 train-test split ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_seed)
    )

    # ── Fit the linear regression model ──────────────────────
    model  = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # ── Performance metrics ───────────────────────────────────
    st.markdown("#### Model Performance — Train 80% / Test 20%")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score",     f"{r2:.4f}")
    mc2.metric("MSE",          f"{mse:.4f}")
    mc3.metric("RMSE",         f"{rmse:.4f}")
    mc4.metric("Train / Test", f"{len(X_train)} / {len(X_test)}")
    st.markdown(
        '<div class="highlight">'
        '📌 <b>R²</b>: fraction of GPA variance explained (1.0 = perfect).<br>'
        '📌 <b>RMSE</b>: average prediction error in GPA units.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Coefficients table ────────────────────────────────────
    st.markdown("#### Regression Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Coefficient": model.coef_,
        "Abs Effect":  np.abs(model.coef_)
    }).sort_values("Abs Effect", ascending=False)
    coef_df["Direction"]   = coef_df["Coefficient"].apply(
        lambda v: "↑ Positive Effect" if v > 0 else "↓ Negative Effect"
    )
    coef_df["Coefficient"] = coef_df["Coefficient"].round(4)
    coef_df["Abs Effect"]  = coef_df["Abs Effect"].round(4)
    st.dataframe(coef_df, use_container_width=True)
    st.markdown(f"**Intercept (β₀):** `{model.intercept_:.4f}`")

    # Display the full regression equation
    equation_terms = " + ".join(
        [f"({model.coef_[i]:.4f} × {FEATURES[i]})" for i in range(len(FEATURES))]
    )
    st.markdown(
        f'<div class="highlight">'
        f'📐 <b>Equation:</b> GPA = {equation_terms} + {model.intercept_:.4f}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Diagnostic plots ──────────────────────────────────────
    st.markdown("#### Regression Diagnostics")
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Actual vs Predicted GPA")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.scatter(y_test, y_pred, alpha=0.55, color="#60a5fa", s=32)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color="#f87171", linestyle="--", linewidth=1.8, label="Perfect prediction")
        ax.set_xlabel(f"Actual {TARGET}")
        ax.set_ylabel(f"Predicted {TARGET}")
        ax.set_title(f"Actual vs Predicted  (R²={r2:.3f})")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.caption("Residuals vs Fitted Values")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.scatter(y_pred, residuals, alpha=0.55, color="#fb923c", s=32)
        ax.axhline(0, color="#f87171", linestyle="--", linewidth=1.8, label="Zero residual line")
        ax.set_xlabel(f"Predicted {TARGET}")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title("Residuals vs Fitted Values")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Live prediction panel ─────────────────────────────────
    st.markdown(f"#### 🔮 Predict a Student's {TARGET}")
    st.caption("Fill in student details and click Predict.")

    inputs = {}
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        if "Age" in FEATURES:
            inputs["Age"] = st.number_input("🎂 Age", min_value=17, max_value=25, value=20)
        if "StudyHoursPerWeek" in FEATURES:
            inputs["StudyHoursPerWeek"] = st.number_input(
                "📚 Study Hours Per Week", min_value=0.0, max_value=40.0, value=15.0, step=0.5
            )
        if "AttendanceRate" in FEATURES:
            inputs["AttendanceRate"] = st.number_input(
                "🏫 Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
            )

    with pc2:
        if "SocialMediaUsage" in FEATURES:
            inputs["SocialMediaUsage"] = st.number_input(
                "📱 Social Media (hrs/day)", min_value=0.0, max_value=12.0, value=3.0, step=0.5
            )
        if "PhysicalActivity" in FEATURES:
            inputs["PhysicalActivity"] = st.number_input(
                "🏃 Physical Activity (hrs/wk)", min_value=0.0, max_value=20.0, value=5.0, step=0.5
            )

    with pc3:
        if "SleepHours" in FEATURES:
            inputs["SleepHours"] = st.number_input(
                "😴 Sleep Hours (night)", min_value=3.0, max_value=12.0, value=7.0, step=0.5
            )
        if "MentalHealthScore" in FEATURES:
            inputs["MentalHealthScore"] = st.number_input(
                "🧠 Mental Health Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5
            )

    if st.button(f"🎯 Predict {TARGET}", use_container_width=True):
        input_row = pd.DataFrame([{f: inputs[f] for f in FEATURES if f in inputs}])
        predicted = float(np.clip(model.predict(input_row)[0], 0.0, 4.0))
        pct       = predicted / 4.0 * 100

        # Classify result into performance band
        if pct >= 85:
            label, color = "Excellent 🏆", "success"
        elif pct >= 70:
            label, color = "Good 👍", "success"
        elif pct >= 55:
            label, color = "Average 📘", "warning"
        else:
            label, color = "Needs Help ⚠️", "error"

        getattr(st, color)(
            f"**Predicted {TARGET}: {predicted:.2f} / 4.0**  —  {label}  ({pct:.1f}%)"
        )

        percentile = (df[TARGET] <= predicted).mean() * 100
        st.markdown(
            f'<div class="highlight">'
            f'📊 This student is predicted to perform better than '
            f'<b>{percentile:.1f}%</b> of students in the dataset.'
            f'</div>',
            unsafe_allow_html=True
        )