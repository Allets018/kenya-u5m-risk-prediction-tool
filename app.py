import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Under-Five Mortality Risk Assessment",
    layout="wide"
)


@st.cache_resource
def load_files():
    model = joblib.load("best_model.joblib")
    schema = joblib.load("input_schema.joblib")
    return model, schema


model, schema = load_files()

st.title("Under-Five Mortality Risk Assessment Tool")

st.write(
    "Enter the maternal, child and household characteristics "
    "to estimate the probability of under-five mortality."
)

inputs = {}

with st.form("risk_form"):

    st.subheader("Maternal characteristics")

    inputs["Maternal Age at 1st Birth"] = st.number_input(
        "Maternal age at first birth",
        min_value=schema["continuous"]["Maternal Age at 1st Birth"]["minimum"],
        max_value=schema["continuous"]["Maternal Age at 1st Birth"]["maximum"],
        value=schema["continuous"]["Maternal Age at 1st Birth"]["default"]
    )

    inputs["Maternal Education"] = st.selectbox(
        "Maternal education",
        schema["categorical"]["Maternal Education"]
    )

    inputs["Maternal Health Status"] = st.selectbox(
        "Maternal health status",
        schema["categorical"]["Maternal Health Status"]
    )

    inputs["Marital Status"] = st.selectbox(
        "Marital status",
        schema["categorical"]["Marital Status"]
    )

    inputs["Antenatal Care Visits"] = st.number_input(
        "Number of antenatal care visits",
        min_value=schema["continuous"]["Antenatal Care Visits"]["minimum"],
        max_value=schema["continuous"]["Antenatal Care Visits"]["maximum"],
        value=schema["continuous"]["Antenatal Care Visits"]["default"]
    )

    st.subheader("Child factors")

    inputs["Preceding Birth Interval"] = st.number_input(
        "Preceding birth interval",
        min_value=schema["continuous"]["Preceding Birth Interval"]["minimum"],
        max_value=schema["continuous"]["Preceding Birth Interval"]["maximum"],
        value=schema["continuous"]["Preceding Birth Interval"]["default"]
    )

    inputs["Birth Weight"] = st.number_input(
        "Birth weight",
        min_value=schema["continuous"]["Birth Weight"]["minimum"],
        max_value=schema["continuous"]["Birth Weight"]["maximum"],
        value=schema["continuous"]["Birth Weight"]["default"]
    )

    inputs["Birth Order"] = st.selectbox(
        "Birth order",
        schema["categorical"]["Birth Order"]
    )

    inputs["Child is Twin"] = st.selectbox(
        "Multiple-birth status",
        schema["categorical"]["Child is Twin"]
    )

    inputs["Child Sex"] = st.selectbox(
        "Child sex",
        schema["categorical"]["Child Sex"]
    )

    inputs["Birth Assistance"] = st.selectbox(
        "Birth assistance",
        schema["categorical"]["Birth Assistance"]
    )

    st.subheader("Household factors")

    inputs["Wealth Index"] = st.selectbox(
        "Household wealth index",
        schema["categorical"]["Wealth Index"]
    )

    inputs["Residence"] = st.selectbox(
        "Residence",
        schema["categorical"]["Residence"]
    )

    submitted = st.form_submit_button("Estimate mortality risk")


if submitted:

    child_profile = pd.DataFrame(
        [inputs],
        columns=schema["column_order"]
    )

    probability = float(
        model.predict_proba(child_profile)[0, 1]
    )

    prediction = int(
        model.predict(child_profile)[0]
    )

    # Temporary thresholds for testing the interface only
    if probability < 0.20:
        risk_category = "Low risk"
    elif probability < 0.50:
        risk_category = "Moderate risk"
    else:
        risk_category = "High risk"

    st.subheader("Prediction results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Risk classification",
        risk_category
    )

    col3.metric(
        "Predicted outcome",
        "Death" if prediction == 1 else "Survival"
    )

    st.dataframe(
        child_profile,
        hide_index=True
    )

    st.warning(
        "This is a study-specific predictive screening tool. "
        "It does not provide a clinical diagnosis and should not "
        "replace professional assessment."
    )
