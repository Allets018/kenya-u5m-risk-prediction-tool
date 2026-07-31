# Corrected Streamlit deployment and sidebar-assistant version
from pathlib import Path
import re

import dalex as dx
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
MODEL_DIR = BASE_DIR

# The saved artifacts are stored beside app.py in the GitHub repository.

MODEL_COLUMN_MAP = {
    "maternal_age_first_birth": "Maternal Age at 1st Birth",
    "birth_interval": "Preceding Birth Interval",
    "anc_visits": "Antenatal Care Visits",
    "birth_weight": "Birth Weight",
    "maternal_education": "Maternal Education",
    "maternal_health_status": "Maternal Health Status",
    "wealth_index": "Wealth Index",
    "birth_order": "Birth Order",
    "multiple_birth": "Child is Twin",
    "child_sex": "Child Sex",
    "birth_assistance": "Birth Assistance",
    "residence": "Residence",
    "marital_status": "Marital Status"
}

APP_COLUMN_MAP = {
    model_name: app_name
    for app_name, model_name in MODEL_COLUMN_MAP.items()
}

BACKGROUND_CATEGORY_MAP = {
    "maternal_health_status": {
        "Bad": "Poor",
        "Very bad": "Poor"
    },
    "multiple_birth": {
        "1st of multiple": "Twin birth",
        "2nd of multiple": "Twin birth"
    },
    "marital_status": {
        "Divorced": "Not married",
        "Living with partner": "Not married",
        "Never in union": "Not married",
        "No longer living together/separated": "Not married",
        "Widowed": "Not married"
    }
}


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
    }

    metadata = joblib.load(
        MODEL_DIR / "app_metadata.joblib"
    )

    background = joblib.load(
        MODEL_DIR / "dalex_background.joblib"
    )

    # Use the categories stored inside the fitted pipeline as the
    # authoritative app options. This prevents stale metadata categories
    # from being sent to the model as unknown values.
    reference_model = next(
        (
            model for model_name, model in models.items()
            if "xgboost" in model_name.lower()
        ),
        next(iter(models.values()))
    )
    preprocessor = reference_model.named_steps["pre_smote"]
    categorical_encoder = preprocessor.named_transformers_["categorical"]

    categorical_columns = next(
        columns
        for name, transformer, columns in preprocessor.transformers_
        if name == "categorical"
    )

    for model_column, categories in zip(
        categorical_columns,
        categorical_encoder.categories_
    ):
        app_column = APP_COLUMN_MAP[model_column]
        metadata["categorical"][app_column] = [
            value.item() if isinstance(value, np.generic) else value
            for value in categories
            if not pd.isna(value)
        ]

    return models, metadata, background


try:
    models, metadata, background = load_artifacts()

except FileNotFoundError as error:
    st.error(
        "One or more required model or metadata files were not "
        "found beside app.py in the repository root."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The saved models could not be loaded. Ensure that "
        "the app uses the same Python environment and package "
        "versions used during model training."
    )
    with st.expander("Model-loading diagnostic details"):
        st.code(f"{type(error).__name__}: {error}")
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


def get_model_categorical_columns(model):
    """Return categorical columns recorded in the fitted preprocessor."""

    preprocessor = model.named_steps["pre_smote"]

    return list(next(
        columns
        for name, transformer, columns in preprocessor.transformers_
        if name == "categorical"
    ))


def prepare_model_input(values, model):
    """Create one row with the exact names and dtypes used in training."""

    data = pd.DataFrame([values]).rename(
        columns=MODEL_COLUMN_MAP
    )

    expected_columns = list(model.feature_names_in_)
    missing_columns = [
        column for column in expected_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing model columns: " + ", ".join(missing_columns)
        )

    data = data.reindex(columns=expected_columns)

    # Birth Order is categorical in the fitted OrdinalEncoder. Casting all
    # categorical inputs to object avoids numeric/object dtype conflicts.
    for column in get_model_categorical_columns(model):
        data[column] = data[column].astype(object)

    return data


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

for variable, replacements in BACKGROUND_CATEGORY_MAP.items():
    if variable in background.columns:
        background[variable] = background[variable].replace(
            replacements
        )

background = background.rename(columns=MODEL_COLUMN_MAP)

reference_model = next(
    (
        model for model_name, model in models.items()
        if "xgboost" in model_name.lower()
    ),
    next(iter(models.values()))
)
background = background.reindex(
    columns=reference_model.feature_names_in_
)

for variable in get_model_categorical_columns(reference_model):
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
default_model = next(
    (
        model_name for model_name in model_names
        if "xgboost" in model_name.lower()
    ),
    model_names[0]
)
default_index = model_names.index(default_model)

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

    try:
        input_data = prepare_model_input(
            user_values,
            selected_model
        )

        selected_probability = float(
            selected_model.predict_proba(input_data)[0, 1]
        )

        risk_category = classify_risk(
            selected_probability,
            thresholds
        )

    except Exception as error:
        st.error(
            "The prediction could not be completed. Check that "
            "the entered values and saved model pipeline use "
            "the same variable names and categories."
        )

        with st.expander("Prediction diagnostic details"):
            st.code(f"{type(error).__name__}: {error}")

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

        model_input = prepare_model_input(
            user_values,
            model
        )

        probability = float(
            model.predict_proba(model_input)[0, 1]
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
# FREE BUILT-IN STUDY KNOWLEDGE ASSISTANT
# ---------------------------------------------------------

MAX_QUESTION_LENGTH = 1000


def build_study_knowledge_base(selected_model_name, thresholds):
    """Return the app's curated study explanations."""

    return [
        {
            "topic": "Purpose of the study",
            "keywords": (
                "purpose", "aim", "objective", "study about",
                "what is this study", "research"
            ),
            "answer": (
                "This study develops a machine-learning research tool "
                "for estimating under-five mortality risk in Kenya. It "
                "compares Logistic Regression, Support Vector Machine, "
                "Random Forest and XGBoost, then uses DALEX to explain "
                "predictions. The application is an educational research "
                "prototype and is not a clinically validated service."
            )
        },
        {
            "topic": "Outcome variable",
            "keywords": (
                "outcome", "target", "class 1", "class 0",
                "death", "survival", "child alive"
            ),
            "answer": (
                "The outcome is under-five mortality. Class 1 represents "
                "death before age five, while class 0 represents survival. "
                "The models estimate the probability of class 1."
            )
        },
        {
            "topic": "Predictor variables",
            "keywords": (
                "predictor", "predictors", "feature", "features",
                "variables", "input"
            ),
            "answer": (
                "The tool uses 13 predictors: maternal age at first birth, "
                "preceding birth interval, antenatal care visits, birth "
                "weight, maternal education, maternal health status, birth "
                "order, household wealth index, multiple-birth status, "
                "child sex, birth assistance, residence and marital status."
            )
        },
        {
            "topic": "Recall",
            "keywords": (
                "recall", "sensitivity", "false negative",
                "miss death", "miss high risk"
            ),
            "answer": (
                "Recall is the proportion of actual under-five deaths that "
                "the model correctly identifies. It was treated as the most "
                "important metric because a false negative means failing to "
                "identify a mortality case. High recall should still be "
                "considered together with precision, specificity and "
                "ROC-AUC."
            )
        },
        {
            "topic": "Precision",
            "keywords": (
                "precision", "false positive", "positive predictive"
            ),
            "answer": (
                "Precision is the proportion of children predicted as "
                "mortality cases who actually belong to the death class. "
                "Low precision means more surviving children are incorrectly "
                "flagged as high risk."
            )
        },
        {
            "topic": "Specificity",
            "keywords": (
                "specificity", "true negative", "survivors identified"
            ),
            "answer": (
                "Specificity is the proportion of actual survivors that the "
                "model correctly identifies as class 0. It shows how well "
                "the model avoids incorrectly flagging survivors as deaths."
            )
        },
        {
            "topic": "Accuracy",
            "keywords": ("accuracy", "correct predictions"),
            "answer": (
                "Accuracy is the proportion of all observations classified "
                "correctly. Because mortality data can be imbalanced, "
                "accuracy alone can be misleading and should not be used as "
                "the only model-selection metric."
            )
        },
        {
            "topic": "F1-score",
            "keywords": ("f1", "f1 score", "harmonic mean"),
            "answer": (
                "The F1-score is the harmonic mean of precision and recall. "
                "It is useful when both missed mortality cases and false "
                "alarms matter."
            )
        },
        {
            "topic": "ROC-AUC",
            "keywords": (
                "roc", "auc", "roc auc", "receiver operating",
                "area under"
            ),
            "answer": (
                "ROC-AUC measures how well a model separates mortality cases "
                "from survivors across all possible classification "
                "thresholds. A value closer to 1 indicates better ranking "
                "ability; 0.5 is approximately random discrimination."
            )
        },
        {
            "topic": "Confusion matrix",
            "keywords": (
                "confusion matrix", "true positive", "false negative",
                "false positive", "true negative"
            ),
            "answer": (
                "The confusion matrix reports true positives, false "
                "positives, true negatives and false negatives. Here, a "
                "true positive is a death correctly identified, while a "
                "false negative is a death incorrectly predicted as "
                "survival."
            )
        },
        {
            "topic": "Cohen's kappa",
            "keywords": ("kappa", "cohen"),
            "answer": (
                "Cohen's kappa measures agreement between predictions and "
                "observed outcomes after accounting for agreement expected "
                "by chance. Higher values indicate stronger agreement."
            )
        },
        {
            "topic": "SMOTENC",
            "keywords": (
                "smotenc", "smote", "imbalance", "class imbalance",
                "oversampling", "synthetic"
            ),
            "answer": (
                "SMOTENC addresses class imbalance when data contain both "
                "numeric and categorical predictors. It creates synthetic "
                "minority-class training observations while respecting "
                "categorical features. It was applied only inside each "
                "training pipeline, never to the test data."
            )
        },
        {
            "topic": "Cross-validation",
            "keywords": (
                "cross validation", "cross-validation", "five fold",
                "5 fold", "repeated stratified", "validation"
            ),
            "answer": (
                "Repeated stratified 5-fold cross-validation was used. Each "
                "run divided the training data into five folds while "
                "preserving the class proportions. Every fold served as the "
                "validation fold, and repeating the process produced a more "
                "stable estimate of model performance."
            )
        },
        {
            "topic": "Hyperparameter tuning",
            "keywords": (
                "hyperparameter", "tuning", "grid search",
                "gridsearch", "best parameters"
            ),
            "answer": (
                "Hyperparameter tuning searches candidate model settings "
                "using cross-validation. The search was refitted primarily "
                "using recall because identifying mortality cases was the "
                "main priority. Tuning used only the training data."
            )
        },
        {
            "topic": "Logistic Regression",
            "keywords": (
                "logistic", "logistic regression", "baseline",
                "benchmark"
            ),
            "answer": (
                "Logistic Regression was the baseline model used as a "
                "benchmark. It estimates mortality probability through a "
                "logistic relationship and is comparatively easy to "
                "interpret."
            )
        },
        {
            "topic": "Support Vector Machine",
            "keywords": (
                "svm", "support vector", "support vector machine",
                "rbf", "kernel"
            ),
            "answer": (
                "The Support Vector Machine uses a decision boundary to "
                "separate mortality cases from survivors. With an RBF "
                "kernel it can represent nonlinear patterns. Probability "
                "estimation was enabled so the app can report mortality "
                "risk probabilities."
            )
        },
        {
            "topic": "Random Forest",
            "keywords": (
                "random forest", "forest", "decision trees",
                "ensemble trees"
            ),
            "answer": (
                "Random Forest combines predictions from many decision "
                "trees. It can model nonlinear relationships and "
                "interactions and is generally less sensitive than a single "
                "tree to random variation in the training data."
            )
        },
        {
            "topic": "XGBoost and model selection",
            "keywords": (
                "xgboost", "xgb", "best model", "selected model",
                "which model", "model selection", "overall performance",
                "performed best", "best performing"
            ),
            "answer": (
                "The tuned XGBoost model was selected for the main DALEX "
                "analysis because it had the best overall balance of the "
                "reported performance measures, with recall remaining a "
                "priority. Model selection was based on cross-validation "
                "results rather than the final test results."
            )
        },
        {
            "topic": "Currently selected model",
            "keywords": (
                "current model", "chosen in app", "selected in app",
                "which model am i using"
            ),
            "answer": (
                f"The model currently selected in the application is "
                f"{selected_model_name}. The model selector allows users "
                "to compare predictions from all four tuned models."
            )
        },
        {
            "topic": "DALEX",
            "keywords": (
                "dalex", "explainable ai", "explainability",
                "interpretability", "explanation"
            ),
            "answer": (
                "DALEX is the model-agnostic explainability framework used "
                "in this study. Global analyses include permutation "
                "variable importance and partial dependence profiles. "
                "Individual analyses include Breakdown and SHAP plots. "
                "These explanations describe model behaviour, not causal "
                "effects."
            )
        },
        {
            "topic": "Variable importance",
            "keywords": (
                "variable importance", "feature importance",
                "permutation importance", "important variable"
            ),
            "answer": (
                "DALEX permutation variable importance measures how much "
                "model performance worsens when a predictor is randomly "
                "permuted. A larger increase in loss indicates greater "
                "predictive importance. Importance does not prove that the "
                "variable causes mortality."
            )
        },
        {
            "topic": "Partial dependence",
            "keywords": (
                "partial dependence", "pdp", "model profile",
                "dependence profile"
            ),
            "answer": (
                "A partial dependence profile shows how the model's average "
                "predicted mortality probability changes as one predictor "
                "changes, while averaging over the other observations. "
                "Correlated predictors can make these profiles harder to "
                "interpret."
            )
        },
        {
            "topic": "Breakdown plot",
            "keywords": (
                "breakdown", "break down", "contribution plot",
                "individual explanation"
            ),
            "answer": (
                "A DALEX Breakdown plot starts from the model's average "
                "prediction and adds each predictor's contribution for one "
                "profile. Positive contributions increase predicted risk "
                "and negative contributions decrease it. The contributions "
                "are model explanations rather than causal effects."
            )
        },
        {
            "topic": "SHAP",
            "keywords": ("shap", "shapley"),
            "answer": (
                "SHAP assigns feature contributions by considering "
                "different orders in which predictors can enter an "
                "explanation. It helps show which values increased or "
                "decreased an individual model prediction."
            )
        },
        {
            "topic": "Risk probability and thresholds",
            "keywords": (
                "risk", "probability", "threshold", "low risk",
                "moderate risk", "high risk", "percent"
            ),
            "answer": (
                f"The predicted probability is the model's estimated chance "
                f"of class 1 for the entered profile. The current research "
                f"categories are low risk below {thresholds['low']:.0%}, "
                f"moderate risk from {thresholds['low']:.0%} to below "
                f"{thresholds['high']:.0%}, and high risk at "
                f"{thresholds['high']:.0%} or above. These thresholds have "
                "not been clinically validated and do not guarantee an "
                "individual outcome."
            )
        },
        {
            "topic": "How the application works",
            "keywords": (
                "how app works", "how tool works", "use the app",
                "make prediction", "prediction process"
            ),
            "answer": (
                "Choose a tuned model, enter all requested maternal, child "
                "and household values, then select Predict mortality risk. "
                "The saved pipeline applies the same preprocessing used "
                "during training and returns a mortality probability, a "
                "research risk category and an individual DALEX Breakdown "
                "explanation."
            )
        },
        {
            "topic": "Limitations and clinical use",
            "keywords": (
                "limitation", "clinical", "diagnosis", "treatment",
                "medical advice", "validated", "replace doctor"
            ),
            "answer": (
                "This is a research and educational prototype, not a "
                "clinical diagnostic system. It must not replace assessment "
                "by a qualified healthcare professional. Predictions depend "
                "on the study data, selected predictors, preprocessing and "
                "model assumptions, and external clinical validation is "
                "still required."
            )
        },
        {
            "topic": "Privacy",
            "keywords": (
                "privacy", "personal information", "data stored",
                "medical record", "name", "identification"
            ),
            "answer": (
                "The application does not intentionally store submitted "
                "profiles, and this built-in assistant sends no questions "
                "to an external AI service. Users should still avoid "
                "entering names, identification numbers, addresses or "
                "medical-record details."
            )
        },
        {
            "topic": "Assistant capabilities",
            "keywords": (
                "help", "what can you answer", "what can you do",
                "topics", "questions"
            ),
            "answer": (
                "I can explain the study purpose, outcome, predictors, "
                "SMOTENC, repeated stratified cross-validation, evaluation "
                "metrics, the four models, tuning, model selection, risk "
                "thresholds and DALEX explanations. Try asking: What does "
                "recall mean in this study?"
            )
        }
    ]


def answer_study_question(question, selected_model_name, thresholds):
    """Match a question to the most relevant curated explanation."""

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        question.lower()
    )
    normalized = " ".join(normalized.split())
    words = set(normalized.split())

    urgent_phrases = (
        "medical emergency",
        "difficulty breathing",
        "not breathing",
        "unconscious",
        "seizure",
        "severe bleeding",
        "child is dying"
    )

    if any(phrase in normalized for phrase in urgent_phrases):
        return (
            "This research assistant cannot assess an emergency. Please "
            "contact a qualified healthcare professional or appropriate "
            "local emergency medical service immediately."
        )

    knowledge_base = build_study_knowledge_base(
        selected_model_name,
        thresholds
    )

    best_entry = None
    best_score = 0

    for entry in knowledge_base:
        score = 0

        for keyword in entry["keywords"]:
            keyword_normalized = keyword.replace("-", " ")

            if " " in keyword_normalized:
                if keyword_normalized in normalized:
                    score += 3 + len(keyword_normalized.split())
            elif keyword_normalized in words:
                score += 2

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None:
        return (
            "That question is not covered by the built-in study knowledge "
            "base. I can explain the study purpose, predictors, SMOTENC, "
            "cross-validation, evaluation metrics, Logistic Regression, "
            "SVM, Random Forest, XGBoost, risk thresholds and DALEX. Please "
            "rephrase your question using one of these topics."
        )

    return best_entry["answer"]


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


with st.sidebar:

    st.markdown("---")

    with st.expander(
        "Free Study Knowledge Assistant",
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

        st.success(
            "This assistant uses a built-in study knowledge base. "
            "It requires no API key, subscription or external AI service."
        )

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
                placeholder=(
                    "Example: What does recall mean in this study?"
                ),
                max_chars=MAX_QUESTION_LENGTH,
                height=90
            )

            ask_assistant = st.form_submit_button(
                "Ask study assistant",
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

            else:

                assistant_answer = answer_study_question(
                    question,
                    selected_model_name,
                    thresholds
                )

                st.session_state.assistant_messages.extend([
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": assistant_answer
                    }
                ])

                st.rerun()

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
