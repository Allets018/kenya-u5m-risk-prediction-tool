from pathlib import Path

import dalex as dx
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib
import uuid

from openai import OpenAI

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Under-Five Mortality Risk Tool",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# 2. APPLICATION STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F6FAF9;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #0B4F4A;
        font-size: 2.7rem !important;
        font-weight: 750 !important;
    }

    h2, h3 {
        color: #12645E;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #CFE3E0;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(15, 118, 110, 0.07);
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #CFE3E0;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.06);
    }

    div[data-baseweb="select"] > div {
        background-color: #F1F7F6;
        border-color: #B8D6D2;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #F1F7F6;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #0F766E;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 650;
        padding: 0.65rem 1.2rem;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #0B5F59;
        color: #FFFFFF;
        border: none;
    }

    [data-testid="stSidebar"] {
        background-color: #E7F2F0;
        border-right: 1px solid #C5DDDA;
    }

    .footer {
        color: #597573;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 3rem;
        border-top: 1px solid #D5E5E3;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# 3. FILE LOCATIONS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Local development stores artifacts in models/. Streamlit Cloud may
# store them beside app.py when uploaded through the GitHub interface.
if not (MODEL_DIR / "logistic_tuned.joblib").exists():
    MODEL_DIR = BASE_DIR


# ---------------------------------------------------------
# 4. LOAD MODELS AND METADATA
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "SVM": joblib.load(
            MODEL_DIR / "svm.joblib"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),
        "XGBoost": joblib.load(
            MODEL_DIR / "xgboost.joblib"
        )
    

    metadata = joblib.load(
        MODEL_DIR / "app_metadata.joblib"
    )

    background = joblib.load(
        MODEL_DIR / "dalex_background.joblib"
    )

    return models, metadata, background


try:
    models, metadata, background = load_artifacts()

except FileNotFoundError as error:
    st.error(
        "One or more required model files were not found "
        "inside the models folder."
    )
    st.code(str(error))
    st.stop()

except Exception:
    st.error(
        "The saved models could not be loaded. Ensure that "
        "the app uses the same Python environment and package "
        "versions used during model training."
    )
    st.stop()


# ---------------------------------------------------------
# 5. DISPLAY LABELS
# ---------------------------------------------------------

LABELS = {
    "maternal_age_first_birth":
        "Maternal age at first birth (years)",

    "birth_interval":
        "Preceding birth interval (months)",

    "anc_visits":
        "Number of antenatal care visits",

    "birth_weight":
        "Birth weight (grams)",

    "maternal_education":
        "Maternal education",

    "maternal_health_status":
        "Maternal health status",

    "birth_order":
        "Birth order",

    "wealth_index":
        "Household wealth index",

    "multiple_birth":
        "Multiple birth status",

    "child_sex":
        "Child sex",

    "birth_assistance":
        "Birth assistance",

    "residence":
        "Place of residence",

    "marital_status":
        "Marital status"
}


# ---------------------------------------------------------
# 6. CATEGORY ORDER
# ---------------------------------------------------------

CATEGORY_ORDER = {
    "maternal_education": [
        "No education",
        "Primary",
        "Secondary",
        "Higher"
    ],

    "maternal_health_status": [
        "Very bad",
        "Bad",
        "Moderate",
        "Good",
        "Very good"
    ],

    "birth_order": (
        list(range(1, 15))
        + [str(value) for value in range(1, 15)]
    ),

    "wealth_index": [
        "Poorest",
        "Poorer",
        "Middle",
        "Richer",
        "Richest"
    ],

    "multiple_birth": [
        "Single birth",
        "1st of multiple",
        "2nd of multiple"
    ],

    # Nominal variables have no low-to-high ranking
    "child_sex": [
        "Female",
        "Male"
    ],

    "birth_assistance": [
        "No",
        "Yes"
    ],

    "residence": [
        "Rural",
        "Urban"
    ],

    "marital_status": [
        "Never in union",
        "Living with partner",
        "Married",
        "No longer living together/separated",
        "Divorced",
        "Widowed"
    ]
}


# ---------------------------------------------------------
# 7. NUMERIC INPUT CONFIGURATION
# ---------------------------------------------------------

NUMERIC_CONFIG = {
    "maternal_age_first_birth": {
        "step": 1.0,
        "format": "%.0f"
    },

    "birth_interval": {
        "step": 1.0,
        "format": "%.0f"
    },

    "anc_visits": {
        "step": 1.0,
        "format": "%.0f"
    },

    "birth_weight": {
        "step": 10.0,
        "format": "%.0f"
    }
}


# ---------------------------------------------------------
# 8. HELPER FUNCTIONS
# ---------------------------------------------------------

def arrange_categories(variable, available_options):
    """Arrange categories without removing unknown values."""

    preferred_order = CATEGORY_ORDER.get(variable)

    if preferred_order is None:
        return sorted(available_options, key=str)

    ordered = [
        value for value in preferred_order
        if value in available_options
    ]

    remaining = [
        value for value in available_options
        if value not in ordered
    ]

    return ordered + sorted(remaining, key=str)


def mortality_probability(model, data):
    """Return probability for class 1: under-five mortality."""

    return model.predict_proba(data)[:, 1]


def classify_risk(probability, thresholds):
    """Classify probability using the configured thresholds."""

    if probability < thresholds["low"]:
        return "Low risk"

    if probability < thresholds["high"]:
        return "Moderate risk"

    return "High risk"


# ---------------------------------------------------------
# 9. PREPARE DALEX BACKGROUND DATA
# ---------------------------------------------------------

background = background.copy()

for variable in metadata["categorical"]:
    if variable in background.columns:
        background[variable] = (
            background[variable]
            .astype(object)
            .where(background[variable].notna(), np.nan)
        )


# ---------------------------------------------------------
# 10. RISK THRESHOLDS
# ---------------------------------------------------------

thresholds = metadata.get(
    "thresholds",
    {
        "low": 0.20,
        "high": 0.50
    }
)

thresholds["low"] = float(thresholds["low"])
thresholds["high"] = float(thresholds["high"])

if thresholds["low"] >= thresholds["high"]:
    st.error(
        "The low-risk threshold must be smaller than "
        "the high-risk threshold."
    )
    st.stop()


# ---------------------------------------------------------
# 11. APPLICATION HEADER
# ---------------------------------------------------------

st.title("Under-Five Mortality Risk Assessment Tool")

st.caption(
    "Machine learning–based research tool for estimating "
    "under-five mortality risk in Kenya."
)

st.warning(
    "This application is intended for research and educational "
    "purposes only. It must not replace professional clinical "
    "assessment or medical decision-making."
)


# ---------------------------------------------------------
# 12. MODEL SELECTION
# ---------------------------------------------------------

model_names = list(models.keys())
default_model = "Tuned XGBoost"

default_index = (
    model_names.index(default_model)
    if default_model in model_names
    else 0
)

with st.sidebar:

    st.header("Model Settings")

    selected_model_name = st.selectbox(
        "Select prediction model",
        model_names,
        index=default_index,
        help=(
            "Tuned XGBoost was selected as the final model "
            "based on its overall performance. Other models "
            "remain available for comparison."
        )
    )

    st.markdown("---")

    st.subheader("Research thresholds")

    st.write(
        f"**Low risk:** below {thresholds['low']:.0%}"
    )

    st.write(
        f"**Moderate risk:** "
        f"{thresholds['low']:.0%} to "
        f"below {thresholds['high']:.0%}"
    )

    st.write(
        f"**High risk:** "
        f"{thresholds['high']:.0%} or higher"
    )

    st.caption(
        "These thresholds are research classifications and "
        "require formal validation before clinical use."
    )

selected_model = models[selected_model_name]


# ---------------------------------------------------------
# 13. USER INPUT FORM
# ---------------------------------------------------------

with st.form("risk_assessment_form"):

    st.subheader("Child and Maternal Information")

    st.write(
        "Enter the information below and select "
        "**Predict mortality risk**."
    )

    form_columns = st.columns(2)
    user_values = {}

    # Continuous variables
    for index, (variable, settings) in enumerate(
        metadata["continuous"].items()
    ):
        config = NUMERIC_CONFIG.get(
            variable,
            {
                "step": 1.0,
                "format": "%.1f"
            }
        )

        with form_columns[index % 2]:
            user_values[variable] = st.number_input(
                LABELS.get(variable, variable),
                min_value=float(settings["min"]),
                max_value=float(settings["max"]),
                value=float(settings["default"]),
                step=config["step"],
                format=config["format"],
                key=variable
            )

    # Categorical variables
    start_index = len(metadata["continuous"])

    for index, (variable, options) in enumerate(
        metadata["categorical"].items(),
        start=start_index
    ):
        ordered_options = arrange_categories(
            variable,
            options
        )

        with form_columns[index % 2]:
            user_values[variable] = st.selectbox(
                LABELS.get(variable, variable),
                ordered_options,
                key=variable
            )

    submitted = st.form_submit_button(
        "Predict mortality risk",
        type="primary"
    )


# Preserve the latest submitted profile when sidebar widgets rerun the app.
if submitted:
    st.session_state["last_prediction_values"] = user_values.copy()

if "last_prediction_values" in st.session_state:
    user_values = st.session_state["last_prediction_values"]
    submitted = True


# ---------------------------------------------------------
# 14. GENERATE PREDICTIONS
# ---------------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [user_values]
    ).reindex(
        columns=metadata["feature_order"]
    )

    try:
        selected_probability = float(
            selected_model.predict_proba(input_data)[0, 1]
        )

        risk_category = classify_risk(
            selected_probability,
            thresholds
        )

    except Exception:
        st.error(
            "The prediction could not be completed. Check that "
            "the entered values and saved model pipeline use "
            "the same variable names and categories."
        )
        st.stop()

    st.subheader("Prediction Result")

    result_column1, result_column2 = st.columns(2)

    with result_column1:
        st.metric(
            "Predicted mortality probability",
            f"{selected_probability:.1%}"
        )

    with result_column2:
        st.metric(
            "Risk classification",
            risk_category
        )

    if risk_category == "High risk":
        st.error(
            "The entered profile was classified as high risk."
        )

    elif risk_category == "Moderate risk":
        st.warning(
            "The entered profile was classified as moderate risk."
        )

    else:
        st.success(
            "The entered profile was classified as low risk."
        )

    st.caption(
        f"Prediction generated using {selected_model_name}."
    )


    # -----------------------------------------------------
    # 15. COMPARE ALL FOUR MODELS
    # -----------------------------------------------------

    comparison_results = []

    for model_name, model in models.items():

        probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        comparison_results.append({
            "Model": model_name,
            "Mortality probability": probability,
            "Risk": classify_risk(
                probability,
                thresholds
            ),
            "Selected": (
                "Yes"
                if model_name == selected_model_name
                else ""
            )
        })

    comparison = pd.DataFrame(comparison_results)

    st.subheader("Comparison Across All Models")

    comparison_display = comparison.copy()

    comparison_display["Mortality probability"] = (
        comparison_display["Mortality probability"]
        .map(lambda value: f"{value:.1%}")
    )

    st.dataframe(
        comparison_display[
            [
                "Model",
                "Mortality probability",
                "Risk",
                "Selected"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )

    chart_data = comparison.sort_values(
        "Mortality probability",
        ascending=True
    )

    comparison_figure = px.bar(
        chart_data,
        x="Mortality probability",
        y="Model",
        orientation="h",
        color="Risk",
        text="Mortality probability",
        color_discrete_map={
            "Low risk": "#2E7D32",
            "Moderate risk": "#F59E0B",
            "High risk": "#C62828"
        }
    )

    comparison_figure.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside"
    )

    comparison_figure.update_layout(
        xaxis_title="Predicted mortality probability",
        yaxis_title="",
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        legend_title="Risk classification",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=40, t=20, b=20)
    )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True
    )


    # -----------------------------------------------------
    # 16. DALEX BREAKDOWN EXPLANATION
    # -----------------------------------------------------

    st.subheader("Factors Influencing This Prediction")

    st.write(
        "The chart shows how each predictor increased or decreased "
        "the mortality probability generated by the selected model."
    )

    try:
        with st.spinner("Generating the DALEX explanation..."):

            explainer = dx.Explainer(
                model=selected_model,
                data=background,
                predict_function=mortality_probability,
                label=selected_model_name,
                model_type="classification",
                verbose=False
            )

            breakdown = explainer.predict_parts(
                new_observation=input_data,
                type="break_down"
            )

            breakdown_figure = breakdown.plot(
                max_vars=len(metadata["feature_order"]),
                show=False
            )

        st.plotly_chart(
            breakdown_figure,
            use_container_width=True
        )

        st.caption(
            "Positive contributions increase predicted mortality risk, "
            "while negative contributions decrease it. These are model "
            "explanations and do not establish causal relationships."
        )

    except Exception as error:
        st.warning(
            "The prediction was completed, but the DALEX explanation "
            "could not be generated."
        )

        with st.expander("DALEX diagnostic details"):
            st.code(f"{type(error).__name__}: {error}")

# ---------------------------------------------------------
# AI ASSISTANT
# ---------------------------------------------------------

AI_MODEL = "gpt-5.6-luna"
MAX_ASSISTANT_QUESTIONS = 10
MAX_QUESTION_LENGTH = 1000


STUDY_ASSISTANT_INSTRUCTIONS = """
You are the study assistant for a research project about a
machine-learning-based under-five mortality risk assessment tool
for Kenya.

Study context:
- The outcome is under-five mortality.
- Class 1 represents death and class 0 represents survival.
- The models are Logistic Regression, Support Vector Machine,
  Random Forest and XGBoost.
- Class imbalance was handled with SMOTENC within the training
  pipeline.
- Models were evaluated using repeated stratified 5-fold
  cross-validation.
- The tuned XGBoost model was selected based on its overall
  performance.
- DALEX is used for variable importance, partial dependence,
  Breakdown and SHAP explanations.
- The application is a research and educational prototype.

Your role:
1. Explain the study, model-development process, evaluation
   metrics, predictors, SMOTENC, DALEX and application features.
2. Use clear language suitable for healthcare workers,
   policymakers, researchers and students.
3. Explain that recall measures the proportion of mortality cases
   correctly identified.
4. Explain that model importance and DALEX contributions are
   predictive associations, not proof of causation.
5. State clearly when information is not available from the study
   context.
6. Keep answers concise and directly related to the study.

Safety restrictions:
- Do not provide a medical diagnosis or treatment recommendation.
- Do not claim the tool is clinically validated.
- Do not tell users that a predicted probability guarantees death
  or survival.
- Do not request names, identification numbers, addresses or other
  personally identifiable information.
- Do not interpret an individual prediction as a substitute for
  professional clinical assessment.
- If asked for urgent medical advice, direct the user to a
  qualified healthcare professional or appropriate emergency
  service.
- Politely decline requests unrelated to this research study.
"""


def get_openai_key():
    """Read the API key without exposing it in the source code."""

    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I can explain the study methodology, "
                "machine-learning models, evaluation metrics, "
                "predictor variables and DALEX explanations. "
                "Please do not enter personal health information."
            )
        }
    ]


if "assistant_question_count" not in st.session_state:
    st.session_state.assistant_question_count = 0


if "assistant_safety_id" not in st.session_state:
    random_session_id = uuid.uuid4().hex

    st.session_state.assistant_safety_id = hashlib.sha256(
        random_session_id.encode("utf-8")
    ).hexdigest()


with st.sidebar:

    st.markdown("---")

    with st.expander(
        "AI Assistant",
        expanded=False
    ):

        st.caption(
            "Ask about the study, models, evaluation metrics, "
            "predictors or DALEX explanations."
        )

        st.info(
            "Research information only. Do not enter names, "
            "identification numbers, medical records or other "
            "personal health information."
        )

        openai_key = get_openai_key()

        if openai_key is None:

            st.caption(
                "The AI assistant has not been configured. Add "
                "OPENAI_API_KEY through Streamlit Cloud secrets "
                "when you are ready to enable it."
            )

        else:

            assistant_consent = st.checkbox(
                "I understand that this is not a clinical service "
                "and I will not enter personal health information.",
                key="assistant_consent"
            )

            for message in st.session_state.assistant_messages[-6:]:

                speaker = (
                    "You"
                    if message["role"] == "user"
                    else "Assistant"
                )

                st.markdown(f"**{speaker}**")
                st.write(message["content"])

            with st.form(
                "assistant_sidebar_form",
                clear_on_submit=True
            ):

                question = st.text_area(
                    "Question",
                    placeholder="Ask a question about the study",
                    max_chars=MAX_QUESTION_LENGTH,
                    height=90
                )

                ask_assistant = st.form_submit_button(
                    "Ask assistant",
                    use_container_width=True
                )

            if ask_assistant:

                question = question.strip()

                if not assistant_consent:

                    st.warning(
                        "Please confirm the privacy and clinical-use "
                        "statement before asking a question."
                    )

                elif not question:

                    st.warning("Please enter a question.")

                elif (
                    st.session_state.assistant_question_count
                    >= MAX_ASSISTANT_QUESTIONS
                ):

                    st.warning(
                        "The maximum number of assistant questions "
                        "for this session has been reached."
                    )

                else:

                    st.session_state.assistant_messages.append({
                        "role": "user",
                        "content": question
                    })

                    try:
                        client = OpenAI(api_key=openai_key)

                        with st.spinner("Preparing an answer..."):

                            moderation = client.moderations.create(
                                model="omni-moderation-latest",
                                input=question
                            )

                            if moderation.results[0].flagged:

                                assistant_answer = (
                                    "I cannot respond to that request. "
                                    "Please ask a question directly "
                                    "related to the research study."
                                )

                            else:

                                conversation = [
                                    {
                                        "role": message["role"],
                                        "content": message["content"]
                                    }
                                    for message in
                                    st.session_state
                                    .assistant_messages[-8:]
                                ]

                                dynamic_context = f"""
Current application context:
- The model currently selected in the interface is
  {selected_model_name}.
- The research risk categories currently use a low threshold
  of {thresholds['low']:.0%} and a high threshold of
  {thresholds['high']:.0%}.
- These thresholds are not yet clinically validated.
"""

                                response = client.responses.create(
                                    model=AI_MODEL,
                                    instructions=(
                                        STUDY_ASSISTANT_INSTRUCTIONS
                                        + dynamic_context
                                    ),
                                    input=conversation,
                                    reasoning={"effort": "low"},
                                    max_output_tokens=400,
                                    safety_identifier=(
                                        st.session_state
                                        .assistant_safety_id
                                    ),
                                    store=False
                                )

                                assistant_answer = (
                                    response.output_text
                                )

                        st.session_state.assistant_messages.append({
                            "role": "assistant",
                            "content": assistant_answer
                        })

                        st.session_state.assistant_question_count += 1
                        st.rerun()

                    except Exception:

                        st.error(
                            "The study assistant is temporarily "
                            "unavailable. Please try again later."
                        )

            if st.button(
                "Clear conversation",
                key="clear_assistant_conversation",
                use_container_width=True
            ):

                st.session_state.assistant_messages = [
                    {
                        "role": "assistant",
                        "content": (
                            "The conversation has been cleared. "
                            "What would you like to know about "
                            "the study?"
                        )
                    }
                ]

                st.session_state.assistant_question_count = 0
                st.rerun()


# ---------------------------------------------------------
# 17. PRIVACY AND FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        This application does not intentionally store submitted
        profiles. Avoid entering names, identification numbers or
        other personally identifiable information.
        <br><br>
        Under-Five Mortality Risk Assessment Tool — Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)
