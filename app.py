import pickle
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("hgb_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    # Rebuild explainer fresh so it's always numpy-compatible
    artifacts["explainer"] = shap.Explainer(artifacts["model"])
    return artifacts

try:
    artifacts = load_model()

    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    explainer = artifacts["explainer"]

except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

# =========================
# TITLE
# =========================
st.title("🫀 Heart Disease Predictor")

st.markdown(
    "Enter patient information below. "
    "The **HistGradientBoosting** model will estimate "
    "the probability of heart disease."
)

st.divider()

# =========================
# INPUT SECTION
# =========================
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Typical Angina",
            2: "2 - Atypical Angina",
            3: "3 - Non-Anginal Pain",
            4: "4 - Asymptomatic"
        }[x]
    )

    bp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200
    )

with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        options=[0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )

    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        help="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"
    )

    maxhr = st.number_input(
        "Max Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f"
    )

col3, col4 = st.columns(2)

with col3:

    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=[1, 2, 3],
        help="1=Upsloping, 2=Flat, 3=Downsloping"
    )

with col4:

    ca = st.selectbox(
        "Number of Major Vessels (0–3)",
        options=[0, 1, 2, 3]
    )

thal = st.selectbox(
    "Thalassemia",
    options=[3, 6, 7],
    format_func=lambda x: {
        3: "Normal (3)",
        6: "Fixed defect (6)",
        7: "Reversible defect (7)"
    }[x]
)

st.divider()

# =========================
# INPUT MAPPING
# =========================
INPUT_MAP = {
    "Age": age,
    "Sex": sex,
    "Chest pain type": cp,
    "BP": bp,
    "Cholesterol": chol,
    "FBS over 120": fbs,
    "EKG results": restecg,
    "Max HR": maxhr,
    "Exercise angina": exang,
    "ST depression": oldpeak,
    "Slope of ST": slope,
    "Number of vessels fluro": ca,
    "Thallium": thal
}

input_array = np.array(
    [INPUT_MAP.get(col, 0) for col in feature_cols]
).reshape(1, -1)

# =========================
# PREDICTION
# =========================
if st.button(
    "🔍 Predict",
    use_container_width=True,
    type="primary"
):

    # Prediction
    proba = model.predict_proba(input_array)[0]
    pred = model.predict(input_array)[0]

    prob_pos = proba[1]

    # =========================
    # RESULT
    # =========================
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(
            f"**Heart Disease: Presence** "
            f"— risk score {prob_pos:.1%}"
        )
    else:
        st.success(
            f"**Heart Disease: Absence** "
            f"— risk score {prob_pos:.1%}"
        )

    st.progress(float(prob_pos))

    st.caption(
        f"Absence: {proba[0]:.1%}  |  "
        f"Presence: {proba[1]:.1%}"
    )

    # =========================
    # EXPLAINABLE AI
    # =========================
    st.divider()

    st.subheader("🔎 Why did the model predict this?")

    st.markdown(
        "SHAP values show how much each feature "
        "**pushed** the prediction toward "
        "Presence (red) or Absence (blue)."
    )

    try:

        shap_values = explainer(input_array)

        # SHAP BAR PLOT
        plt.figure(figsize=(8, 5))

        shap.plots.bar(
            shap_values[0],
            show=False
        )

        st.pyplot(plt.gcf())

        plt.close()

        # =========================
        # FEATURE CONTRIBUTIONS
        # =========================
        st.subheader("📝 Feature Contributions")

        vals = shap_values.values[0]

        top3 = np.argsort(np.abs(vals))[::-1][:3]

        for rank, idx in enumerate(top3, 1):

            fname = feature_cols[idx]
            fval = input_array[0][idx]
            impact = vals[idx]

            direction = (
                "increased"
                if impact > 0
                else "decreased"
            )

            st.write(
                f"**{rank}. {fname} = {fval}** → "
                f"{direction} the risk "
                f"by `{abs(impact):.3f}`"
            )

    except Exception as e:

        st.warning(
            f"SHAP explanation could not "
            f"be displayed: {e}"
        )

    # =========================
    # INPUT SUMMARY
    # =========================
    with st.expander("📋 Input summary"):

        for col, val in zip(feature_cols, input_array[0]):

            st.write(f"**{col}:** {val}")

# =========================
# FOOTER
# =========================
st.divider()

st.caption(
    "Model: HistGradientBoostingClassifier "
    "· IS411 Data Modelling · Kelompok 7"
)
