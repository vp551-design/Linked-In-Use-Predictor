import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ---------- CLEANING FUNCTION ----------
def clean_sm(x):
    """Return 1 if x == 1 (uses LinkedIn), otherwise 0."""
    return np.where(x == 1, 1, 0)


# ---------- MODEL TRAINING (CACHED) ----------
@st.cache_data
def train_model():
    s = pd.read_csv("social_media_usage.csv")

    # Build the cleaned dataset (Part 1 logic)
    ss = pd.DataFrame({
        "income": np.where(s["income"] > 9, np.nan, s["income"]),
        "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent": np.where(s["par"] == 1, 1, 0),
        "married": np.where(s["marital"] == 1, 1, 0),
        "female": np.where(s["gender"] == 2, 1, 0),
        "age": np.where(s["age"] > 98, np.nan, s["age"]),
        "sm_li": clean_sm(s["web1h"]),
    })

    ss = ss.dropna()

    y = ss["sm_li"]
    X = ss[["income", "education", "parent", "married", "female", "age"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy, ss


model, test_accuracy, ss = train_model()


# -------------------- UI --------------------
st.title("LinkedIn Use Predictor")

st.write(
    """This app uses a logistic regression model (from Part 1 of the project)
    to estimate the probability that a person uses LinkedIn, based on demographic factors."""
)

st.markdown(f"**Model test accuracy:** `{test_accuracy:.3f}`")


# ---------- USER INPUT SECTION ----------
st.header("Enter person information")

income_labels = {
    1: "1 – Less than $10,000",
    2: "2 – $10–20k",
    3: "3 – $20–30k",
    4: "4 – $30–40k",
    5: "5 – $40–50k",
    6: "6 – $50–75k",
    7: "7 – $75–100k",
    8: "8 – $100–150k",
}

educ_labels = {
    1: "1 – Less than high school",
    2: "2 – HS incomplete",
    3: "3 – HS graduate",
    4: "4 – Some college, no degree",
    5: "5 – Associate degree",
    6: "6 – Bachelor's degree",
    7: "7 – Some postgraduate schooling",
    8: "8 – Postgraduate / professional degree",
}

income = st.selectbox("Household income", list(income_labels.keys()),
                      format_func=lambda x: income_labels[x])

education = st.selectbox("Education level", list(educ_labels.keys()),
                         format_func=lambda x: educ_labels[x])

parent = 1 if st.radio("Parent of child <18?", ["No", "Yes"]) == "Yes" else 0
married = 1 if st.radio("Marital status", ["Not married", "Married"]) == "Married" else 0
female = 1 if st.radio("Gender", ["Male", "Female"]) == "Female" else 0

age = st.slider("Age", 18, 95, 30)


# ---------- MAKE PREDICTION ----------
if st.button("Predict LinkedIn Use"):
    user_data = pd.DataFrame({
        "income": [income],
        "education": [education],
        "parent": [parent],
        "married": [married],
        "female": [female],
        "age": [age],
    })

    prob = model.predict_proba(user_data)[0][1]
    pred = model.predict(user_data)[0]

    st.subheader("Prediction")
    if pred == 1:
        st.success("This person is **predicted to use LinkedIn.**")
    else:
        st.error("This person is **predicted NOT to use LinkedIn.**")

    st.write(f"Estimated probability: **{prob:.2%}**")
    st.progress(int(prob * 100))


# ---------------------------- VISUALIZATIONS ------------------------------
# -------------------------------------------------------------------------

with st.expander("📊 View Data Visualizations & Insights"):
    st.header("Age Distribution by LinkedIn Use")

    # ---- FIXED AGE BINS WITH CLEAN LABELS ----
    ss_age = ss.copy()
    bins = 15
    ss_age["age_bin"] = pd.cut(ss_age["age"], bins=bins)

    # Clean labels like "20–25"
    ss_age["age_bin_label"] = ss_age["age_bin"].apply(
        lambda x: f"{int(x.left)}–{int(x.right)}" if pd.notnull(x) else ""
    )

    chart1 = alt.Chart(ss_age).mark_bar().encode(
        x=alt.X("age_bin_label:N", title="Age Range"),
        y="count()",
        color=alt.Color("sm_li:N", title="LinkedIn user"),
        tooltip=["age_bin_label", "sm_li", "count()"]
    ).properties(width=700)

    st.altair_chart(chart1, use_container_width=True)

    st.markdown("""
    **Insight:**
    - LinkedIn use peaks between ages **30–55**.  
    - Usage drops sharply after age 60.  
    - Younger adults under 25 also show moderate adoption.
    """)

    # ===== INCOME vs PROBABILITY =====
    st.header("LinkedIn Use Probability by Income Level")

    income_prob = ss.groupby("income")["sm_li"].mean().reset_index()

    chart2 = alt.Chart(income_prob).mark_line(point=True).encode(
        x=alt.X("income:O", title="Income Level"),
        y=alt.Y("sm_li:Q", title="Probability of LinkedIn Use"),
        tooltip=["income", "sm_li"]
    ).properties(width=700)

    st.altair_chart(chart2, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Higher income strongly correlates with higher LinkedIn usage.  
    - People earning **$100k+** show the highest probability.
    """)

    # ===== EDUCATION BAR CHART =====
    st.header("LinkedIn Use by Education Level")

    edu_prob = ss.groupby("education")["sm_li"].mean().reset_index()

    chart3 = alt.Chart(edu_prob).mark_bar().encode(
        x=alt.X("education:O", title="Education Level"),
        y=alt.Y("sm_li:Q", title="Probability of LinkedIn Use"),
        tooltip=["education", "sm_li"]
    ).properties(width=700)

    st.altair_chart(chart3, use_container_width=True)

    st.markdown("""
    **Insight:**
    - LinkedIn use rises steadily with higher education.  
    - The highest usage occurs among those with **postgraduate degrees**.
    """)

