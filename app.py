from pathlib import Path
import html
import json
import re

import dalex as dx
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Under-Five Mortality Risk Assessment",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CLEAN BLACK AND DARK-BLUE THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --black: #05070B;
        --navy: #081C36;
        --blue: #0A2A52;
        --blue-2: #123E70;
        --ink: #111827;
        --muted: #667085;
        --surface: #FFFFFF;
        --background: #F5F6F8;
        --border: #D8DEE8;
    }

    .stApp {
        background: var(--background);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 760 !important;
    }

    h3 {
        font-size: 1.08rem !important;
        font-weight: 720 !important;
    }

    .app-header {
        background: linear-gradient(120deg, #05070B 0%, #081C36 58%, #0A2A52 100%);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 1.45rem 1.7rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 10px 26px rgba(5, 7, 11, 0.16);
    }

    .app-header h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: clamp(1.75rem, 4vw, 2.55rem) !important;
    }

    .app-header p {
        margin: 0.45rem 0 0;
        color: #D7E2F0;
        line-height: 1.55;
        max-width: 850px;
    }

    .section-note {
        color: var(--muted);
        margin-top: -0.3rem;
        margin-bottom: 0.85rem;
    }

    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 5px 18px rgba(8, 28, 54, 0.05);
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(8, 28, 54, 0.045);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 800;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    textarea {
        background-color: #FBFCFE !important;
        border-color: #C9D2DF !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        min-height: 2.65rem;
        font-weight: 700;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--blue);
        color: #FFFFFF;
        border: 1px solid var(--blue);
    }

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--navy);
        border-color: var(--navy);
    }

    [data-testid="stSidebar"] {
        background: #0A1424;
        border-right: 1px solid #172A44;
    }

    [data-testid="stSidebar"] * {
        color: #F4F7FB;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] small {
        color: #B9C7D8 !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 720;
        color: #526071;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--blue);
    }

    .result-panel {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-left: 6px solid var(--blue);
        border-radius: 13px;
        padding: 1.1rem 1.25rem;
        margin-top: 0.7rem;
    }

    .result-panel .probability {
        color: var(--black);
        font-size: 2.35rem;
        font-weight: 840;
        line-height: 1;
        margin: 0.3rem 0 0.55rem;
    }

    .risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.72rem;
        font-size: 0.88rem;
        font-weight: 760;
        color: white;
    }

    .risk-low { background: #315B7D; }
    .risk-moderate { background: #173A5E; }
    .risk-high { background: #05070B; }

    .footer {
        text-align: center;
        color: #4B5563;
        font-size: 0.84rem;
        line-height: 1.55;
        border-top: 1px solid var(--border);
        padding: 1.1rem 0 0.2rem;
        margin-top: 2.6rem;
    }

    .compact-result-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    .method-box {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
    }

    @media (max-width: 780px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        .app-header {
            padding: 1.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS, LABELS AND DISPLAY CATEGORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

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
    "marital_status": "Marital Status",
}

APP_COLUMN_MAP = {
    model_name: app_name
    for app_name, model_name in MODEL_COLUMN_MAP.items()
}

LABELS = {
    "maternal_age_first_birth": "Maternal age at first birth",
    "birth_interval": "Preceding birth interval",
    "anc_visits": "Antenatal care visits",
    "birth_weight": "Birth weight",
    "maternal_education": "Maternal education",
    "maternal_health_status": "Maternal health status",
    "wealth_index": "Household wealth",
    "birth_order": "Birth order",
    "multiple_birth": "Multiple-birth status",
    "child_sex": "Child sex",
    "birth_assistance": "Birth assistance",
    "residence": "Residence",
    "marital_status": "Marital status",
}

NUMERIC_CONFIG = {
    "maternal_age_first_birth": {
        "unit": "years", "step": 1.0, "format": "%.0f"
    },
    "birth_interval": {
        "unit": "months", "step": 1.0, "format": "%.0f"
    },
    "anc_visits": {
        "unit": "visits", "step": 1.0, "format": "%.0f"
    },
    "birth_weight": {
        "unit": "grams", "step": 10.0, "format": "%.0f"
    },
}

# These are the only categories displayed to the user.
DISPLAY_OPTIONS = {
    "maternal_education": [
        "No education", "Primary", "Secondary", "Higher"
    ],
    "maternal_health_status": [
        "Poor", "Moderate", "Good"
    ],
    "wealth_index": [
        "Poor", "Middle", "Rich"
    ],
    "multiple_birth": [
        "Single birth", "Twin birth"
    ],
    "child_sex": [
        "Female", "Male"
    ],
    "birth_assistance": [
        "No", "Yes"
    ],
    "residence": [
        "Rural", "Urban"
    ],
    "marital_status": [
        "Not married", "Married"
    ],
}

# Maps the simplified display categories to categories used by a fitted model.
# Exact matches are always preferred. The alternatives allow the app to remain
# compatible with older saved pipelines that used finer category groupings.
MODEL_VALUE_ALIASES = {
    "maternal_health_status": {
        "Poor": ["Poor", "Bad", "Very bad"],
        "Moderate": ["Moderate"],
        "Good": ["Good", "Very good"],
    },
    "wealth_index": {
        "Poor": ["Poor", "Poorer", "Poorest"],
        "Middle": ["Middle"],
        "Rich": ["Rich", "Richer", "Richest"],
    },
    "multiple_birth": {
        "Single birth": ["Single birth"],
        "Twin birth": ["Twin birth", "1st of multiple", "2nd of multiple"],
    },
    "marital_status": {
        "Not married": [
            "Not married", "Never in union", "Living with partner",
            "No longer living together/separated", "Divorced", "Widowed"
        ],
        "Married": ["Married"],
    },
}

FIELD_GROUPS = {
    "Maternal characteristics": [
        "maternal_age_first_birth",
        "maternal_education",
        "maternal_health_status",
        "anc_visits",
        "marital_status",
    ],
    "Child and birth characteristics": [
        "birth_interval",
        "birth_weight",
        "birth_order",
        "multiple_birth",
        "child_sex",
        "birth_assistance",
    ],
    "Household characteristics": [
        "wealth_index",
        "residence",
    ],
}


# =========================================================
# MODEL LOADING AND PIPELINE INSPECTION
# =========================================================

def get_preprocessor(model):
    if not hasattr(model, "named_steps"):
        raise TypeError("The saved estimator is not a fitted pipeline.")

    for step_name in ("pre_smote", "preprocessor", "preprocess"):
        if step_name in model.named_steps:
            return model.named_steps[step_name]

    raise KeyError("No fitted preprocessing step was found.")


def get_categorical_transformer(preprocessor):
    for name, transformer, columns in preprocessor.transformers_:
        if name in ("categorical", "category", "cat"):
            return transformer, list(columns)

    raise KeyError("No categorical transformer was found.")


def get_encoder(transformer):
    if hasattr(transformer, "categories_"):
        return transformer

    if hasattr(transformer, "named_steps"):
        for candidate in reversed(list(transformer.named_steps.values())):
            if hasattr(candidate, "categories_"):
                return candidate

    raise AttributeError("The fitted categorical encoder has no categories_.")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Support Vector Machine": joblib.load(
            MODEL_DIR / "svm.joblib"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),
        "XGBoost": joblib.load(
            MODEL_DIR / "xgboost.joblib"
        ),
    }

    metadata = joblib.load(MODEL_DIR / "app_metadata.joblib")
    background = joblib.load(MODEL_DIR / "dalex_background.joblib")

    return models, metadata, background


try:
    models, metadata, background_raw = load_artifacts()
except FileNotFoundError as error:
    st.error("A required model or metadata file was not found beside app.py.")
    st.code(str(error))
    st.stop()
except Exception as error:
    st.error(
        "The saved model files could not be loaded. Use the same package "
        "environment used during model training."
    )
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")
    st.stop()


@st.cache_resource(show_spinner=False)
def model_category_map(model_name):
    model = models[model_name]
    preprocessor = get_preprocessor(model)
    categorical_transformer, categorical_columns = (
        get_categorical_transformer(preprocessor)
    )
    encoder = get_encoder(categorical_transformer)

    return {
        column: [
            value.item() if isinstance(value, np.generic) else value
            for value in categories
            if not pd.isna(value)
        ]
        for column, categories in zip(
            categorical_columns,
            encoder.categories_,
        )
    }


def resolve_model_category(app_variable, display_value, model_name):
    model_column = MODEL_COLUMN_MAP[app_variable]
    categories = model_category_map(model_name).get(model_column, [])

    if display_value in categories:
        return display_value

    candidates = MODEL_VALUE_ALIASES.get(
        app_variable,
        {},
    ).get(display_value, [display_value])

    for candidate in candidates:
        if candidate in categories:
            return candidate

    raise ValueError(
        f"The category '{display_value}' for {LABELS[app_variable]} "
        f"is not compatible with the saved {model_name} pipeline. "
        f"Available categories: {categories}"
    )


def get_model_categorical_columns(model):
    preprocessor = get_preprocessor(model)
    _, columns = get_categorical_transformer(preprocessor)
    return columns


def prepare_model_input(values, model_name):
    model = models[model_name]
    resolved_values = values.copy()

    for app_variable in DISPLAY_OPTIONS:
        if app_variable in resolved_values:
            resolved_values[app_variable] = resolve_model_category(
                app_variable,
                resolved_values[app_variable],
                model_name,
            )

    data = pd.DataFrame([resolved_values]).rename(
        columns=MODEL_COLUMN_MAP
    )

    expected_columns = list(model.feature_names_in_)
    data = data.reindex(columns=expected_columns)

    if data.isna().all(axis=0).any():
        missing = data.columns[data.isna().all(axis=0)].tolist()
        raise ValueError("Missing model columns: " + ", ".join(missing))

    for column in get_model_categorical_columns(model):
        data[column] = data[column].astype(object)

    return data


def mortality_probability(model, data):
    return model.predict_proba(data)[:, 1]


# =========================================================
# DALEX BACKGROUND
# =========================================================

background = background_raw.copy().rename(columns=MODEL_COLUMN_MAP)
reference_model_name = "XGBoost"
reference_model = models[reference_model_name]
background = background.reindex(columns=reference_model.feature_names_in_)
reference_categories = model_category_map(reference_model_name)


def normalize_background_category(model_column, value):
    """Map old or fine-grained background labels to valid model categories."""

    if pd.isna(value):
        return value

    available = reference_categories.get(model_column, [])
    if value in available:
        return value

    app_variable = APP_COLUMN_MAP.get(model_column)
    if app_variable is None:
        return value

    aliases = MODEL_VALUE_ALIASES.get(app_variable, {})

    for display_value, candidates in aliases.items():
        if value == display_value or value in candidates:
            return resolve_model_category(
                app_variable,
                display_value,
                reference_model_name,
            )

    return value


for column in get_model_categorical_columns(reference_model):
    background[column] = background[column].map(
        lambda value: normalize_background_category(column, value)
    ).astype(object)


@st.cache_resource(show_spinner=False)
def build_explainer(model_name):
    return dx.Explainer(
        model=models[model_name],
        data=background,
        predict_function=mortality_probability,
        label=model_name,
        model_type="classification",
        verbose=False,
    )


# =========================================================
# RISK THRESHOLDS AND HELPERS
# =========================================================

thresholds = metadata.get(
    "thresholds",
    {"low": 0.20, "high": 0.50},
)
thresholds = {
    "low": float(thresholds["low"]),
    "high": float(thresholds["high"]),
}

if thresholds["low"] >= thresholds["high"]:
    st.error("The low threshold must be smaller than the high threshold.")
    st.stop()


def classify_risk(probability):
    if probability < thresholds["low"]:
        return "Low risk"
    if probability < thresholds["high"]:
        return "Moderate risk"
    return "High risk"


def risk_css_class(risk):
    return {
        "Low risk": "risk-low",
        "Moderate risk": "risk-moderate",
        "High risk": "risk-high",
    }[risk]


def render_input(variable):
    label = LABELS[variable]

    if variable in metadata["continuous"]:
        settings = metadata["continuous"][variable]
        config = NUMERIC_CONFIG[variable]
        return st.number_input(
            f"{label} ({config['unit']})",
            min_value=float(settings["min"]),
            max_value=float(settings["max"]),
            value=float(settings["default"]),
            step=float(config["step"]),
            format=config["format"],
            key=f"input_{variable}",
        )

    if variable == "birth_order":
        categories = model_category_map("XGBoost").get(
            MODEL_COLUMN_MAP[variable],
            [],
        )
        options = sorted(categories, key=lambda value: float(value))
        return st.selectbox(
            label,
            options,
            key=f"input_{variable}",
        )

    return st.selectbox(
        label,
        DISPLAY_OPTIONS[variable],
        key=f"input_{variable}",
    )


def calculate_all_predictions(values):
    results = []

    for model_name, model in models.items():
        input_data = prepare_model_input(values, model_name)
        probability = float(model.predict_proba(input_data)[0, 1])
        results.append({
            "Model": model_name,
            "Mortality probability": probability,
            "Risk": classify_risk(probability),
        })

    return pd.DataFrame(results)


def extract_breakdown_summary(result, limit=5):
    if not isinstance(result, pd.DataFrame):
        return []

    variable_column = next(
        (
            column for column in
            ("variable", "variable_name", "_variable_")
            if column in result.columns
        ),
        None,
    )
    contribution_column = next(
        (
            column for column in
            ("contribution", "_contribution_")
            if column in result.columns
        ),
        None,
    )

    if variable_column is None or contribution_column is None:
        return []

    data = result.copy()
    data = data[
        ~data[variable_column].astype(str).str.startswith("_")
    ]
    data["absolute_contribution"] = data[contribution_column].abs()
    data = data.sort_values(
        "absolute_contribution",
        ascending=False,
    ).head(limit)

    return [
        {
            "variable": str(row[variable_column]),
            "contribution": float(row[contribution_column]),
        }
        for _, row in data.iterrows()
    ]


def generate_breakdown_for_profile(profile, model_name):
    """Generate and store a DALEX breakdown for one submitted profile."""

    input_data = prepare_model_input(profile, model_name)
    explainer = build_explainer(model_name)
    breakdown = explainer.predict_parts(
        new_observation=input_data,
        type="break_down",
    )

    figure = breakdown.plot(
        max_vars=10,
        show=False,
    )
    summary = extract_breakdown_summary(
        breakdown.result,
        limit=5,
    )
    signature = (
        model_name,
        tuple((key, str(value)) for key, value in profile.items()),
    )

    return figure, summary, signature


# =========================================================
# STUDY KNOWLEDGE ASSISTANT
# =========================================================

@st.cache_data(show_spinner=False)
def load_study_knowledge():
    path = BASE_DIR / "study_knowledge.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


STUDY_KNOWLEDGE = load_study_knowledge()


@st.cache_resource(show_spinner=False)
def build_knowledge_index():
    documents = [
        " ".join(
            [
                entry["title"],
                " ".join(entry.get("keywords", [])),
                entry["content"],
            ]
        )
        for entry in STUDY_KNOWLEDGE
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def latest_prediction_context(selected_model_name):
    values = st.session_state.get("last_profile")
    if values is None:
        return None

    comparison = calculate_all_predictions(values)
    row = comparison.loc[
        comparison["Model"] == selected_model_name
    ].iloc[0]

    return {
        "model": selected_model_name,
        "probability": float(row["Mortality probability"]),
        "risk": row["Risk"],
        "contributions": st.session_state.get(
            "breakdown_summary",
            [],
        ),
    }


def answer_study_question(question, selected_model_name):
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = " ".join(normalized.split())

    emergency_phrases = (
        "not breathing", "unconscious", "seizure",
        "severe bleeding", "child is dying", "medical emergency",
    )
    if any(phrase in normalized for phrase in emergency_phrases):
        return (
            "This assistant cannot assess an emergency. Contact an "
            "appropriate emergency medical service or qualified healthcare "
            "professional immediately."
        )

    context = latest_prediction_context(selected_model_name)
    context_phrases = (
        "latest prediction", "my result", "this result",
        "explain the result", "why high risk", "why moderate risk",
        "why low risk", "what does the prediction mean",
    )

    if context is not None and any(
        phrase in normalized for phrase in context_phrases
    ):
        response = (
            f"The latest assessment used {context['model']} and produced "
            f"a predicted probability of {context['probability']:.1%}, "
            f"classified as {context['risk']}."
        )

        contributions = context["contributions"]
        if contributions:
            increasing = [
                item["variable"]
                for item in contributions
                if item["contribution"] > 0
            ][:3]
            decreasing = [
                item["variable"]
                for item in contributions
                if item["contribution"] < 0
            ][:3]

            if increasing:
                response += (
                    " The strongest recorded factors increasing the "
                    "prediction were " + ", ".join(increasing) + "."
                )
            if decreasing:
                response += (
                    " Factors decreasing the prediction included "
                    + ", ".join(decreasing) + "."
                )
        else:
            response += (
                " Generate the DALEX breakdown to see the strongest "
                "increasing and decreasing contributions."
            )

        return response

    vectorizer, matrix = build_knowledge_index()
    query = vectorizer.transform([normalized])
    scores = cosine_similarity(query, matrix).flatten()
    top_indices = scores.argsort()[::-1][:2]

    if scores[top_indices[0]] < 0.025:
        return (
            "The study materials do not provide a sufficiently clear answer "
            "to that question. Ask about the proposal, objectives, dataset, "
            "variables, preprocessing, exploratory results, chi-square tests, "
            "model development, performance metrics, DALEX findings, or the "
            "risk assessment tool."
        )

    best = STUDY_KNOWLEDGE[int(top_indices[0])]["content"]

    # Add a second section only when it is meaningfully related and adds detail.
    if scores[top_indices[1]] >= max(0.06, scores[top_indices[0]] * 0.72):
        second = STUDY_KNOWLEDGE[int(top_indices[1])]["content"]
        if second != best:
            return best + "\n\n" + second

    return best


# =========================================================
# SESSION STATE
# =========================================================

if "last_profile" not in st.session_state:
    st.session_state.last_profile = None
if "breakdown_figure" not in st.session_state:
    st.session_state.breakdown_figure = None
if "breakdown_summary" not in st.session_state:
    st.session_state.breakdown_summary = []
if "breakdown_signature" not in st.session_state:
    st.session_state.breakdown_signature = None
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me about any part of the study, including the proposal, "
                "dataset, variables, analysis, model results, DALEX findings, "
                "or the latest prediction."
            ),
        }
    ]


# =========================================================
# SIDEBAR
# =========================================================

model_names = list(models.keys())

with st.sidebar:
    st.markdown("## Model")
    selected_model_name = st.selectbox(
        "Prediction model",
        model_names,
        index=model_names.index("XGBoost"),
    )

    st.markdown("---")
    st.markdown("## Risk bands")
    st.write(f"Low: below {thresholds['low']:.0%}")
    st.write(
        f"Moderate: {thresholds['low']:.0%}–"
        f"{thresholds['high']:.0%}"
    )
    st.write(f"High: {thresholds['high']:.0%} or above")

    st.markdown("---")
    if st.button("Clear assessment", use_container_width=True):
        st.session_state.last_profile = None
        st.session_state.breakdown_figure = None
        st.session_state.breakdown_summary = []
        st.session_state.breakdown_signature = None
        st.rerun()


# =========================================================
# HEADER AND NAVIGATION
# =========================================================

st.markdown(
    """
    <div class="app-header">
        <h1>Under-Five Mortality Risk Assessment Tool</h1>
        <p>
            Estimate under-five mortality probability using maternal,
            child, birth, and household characteristics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

assessment_tab, comparison_tab, assistant_tab, methodology_tab = st.tabs(
    [
        "Risk assessment",
        "Model comparison",
        "Study assistant",
        "Methodology",
    ]
)


# =========================================================
# RISK ASSESSMENT
# =========================================================

with assessment_tab:
    st.subheader("Enter the child profile")
    st.markdown(
        '<p class="section-note">Complete the maternal, child, birth, and household fields below.</p>',
        unsafe_allow_html=True,
    )

    with st.form("risk_form"):
        user_values = {}

        for group_name, variables in FIELD_GROUPS.items():
            st.markdown(f"### {group_name}")
            columns = st.columns(2)

            for index, variable in enumerate(variables):
                with columns[index % 2]:
                    user_values[variable] = render_input(variable)

            if group_name != "Household characteristics":
                st.markdown("---")

        submitted = st.form_submit_button(
            "Predict mortality risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        st.session_state.last_profile = user_values.copy()
        st.session_state.breakdown_figure = None
        st.session_state.breakdown_summary = []
        st.session_state.breakdown_signature = None

    if st.session_state.last_profile is None:
        st.info("Enter the profile and select **Predict mortality risk**.")

    else:
        try:
            comparison = calculate_all_predictions(
                st.session_state.last_profile
            )
            selected = comparison.loc[
                comparison["Model"] == selected_model_name
            ].iloc[0]
            probability = float(selected["Mortality probability"])
            risk = selected["Risk"]
        except Exception as error:
            st.error("The prediction could not be completed.")
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")
            st.stop()

        st.subheader("Prediction result")
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Predicted probability", f"{probability:.1%}")
        metric_2.metric("Risk category", risk)
        metric_3.metric("Selected model", selected_model_name)

        st.markdown(
            '<p class="compact-result-note">The displayed category is based on the study-specific probability thresholds shown in the sidebar.</p>',
            unsafe_allow_html=True,
        )

        explain_col, spacer_col = st.columns([1, 1])
        with explain_col:
            explain_clicked = st.button(
                "Explain this prediction",
                type="primary",
                use_container_width=True,
            )

        current_signature = (
            selected_model_name,
            tuple(
                (key, str(value))
                for key, value in st.session_state.last_profile.items()
            ),
        )

        if st.session_state.breakdown_signature != current_signature:
            st.session_state.breakdown_figure = None
            st.session_state.breakdown_summary = []
            st.session_state.breakdown_signature = None

        if explain_clicked:
            try:
                with st.spinner("Generating the DALEX explanation..."):
                    (
                        st.session_state.breakdown_figure,
                        st.session_state.breakdown_summary,
                        st.session_state.breakdown_signature,
                    ) = generate_breakdown_for_profile(
                        st.session_state.last_profile,
                        selected_model_name,
                    )
            except Exception as error:
                st.warning("The DALEX explanation could not be generated.")
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")

        if st.session_state.breakdown_figure is not None:
            with st.expander(
                "DALEX explanation for this prediction",
                expanded=True,
            ):
                st.plotly_chart(
                    st.session_state.breakdown_figure,
                    use_container_width=True,
                )
                st.caption(
                    "Positive contributions increase the predicted mortality "
                    "probability, while negative contributions decrease it."
                )

        with st.expander("Review submitted profile", expanded=False):
            profile_table = pd.DataFrame(
                {
                    "Variable": [
                        LABELS.get(key, key)
                        for key in st.session_state.last_profile
                    ],
                    "Entered value": list(
                        st.session_state.last_profile.values()
                    ),
                }
            )
            st.dataframe(
                profile_table,
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# MODEL COMPARISON
# =========================================================

with comparison_tab:
    if st.session_state.last_profile is None:
        st.info("Generate a prediction first to compare the four models.")
    else:
        comparison = calculate_all_predictions(
            st.session_state.last_profile
        )
        comparison["Selected"] = np.where(
            comparison["Model"] == selected_model_name,
            "Yes",
            "",
        )

        st.subheader("Prediction comparison across models")
        st.markdown(
            '<p class="section-note">The table and chart compare the probability assigned to the same submitted profile by each model.</p>',
            unsafe_allow_html=True,
        )

        display_table = comparison.copy()
        display_table["Mortality probability"] = display_table[
            "Mortality probability"
        ].map(lambda value: f"{value:.1%}")

        st.dataframe(
            display_table[
                ["Model", "Mortality probability", "Risk", "Selected"]
            ],
            hide_index=True,
            use_container_width=True,
        )

        chart_data = comparison.sort_values(
            "Mortality probability",
            ascending=True,
        )

        figure = px.bar(
            chart_data,
            x="Mortality probability",
            y="Model",
            orientation="h",
            text="Mortality probability",
            color="Risk",
            color_discrete_map={
                "Low risk": "#315B7D",
                "Moderate risk": "#173A5E",
                "High risk": "#05070B",
            },
        )
        figure.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
        )
        figure.update_layout(
            height=360,
            xaxis_title="Predicted mortality probability",
            yaxis_title="",
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            legend_title="Risk",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=55, t=15, b=20),
        )
        st.plotly_chart(figure, use_container_width=True)


# =========================================================
# COMPREHENSIVE STUDY ASSISTANT
# =========================================================

with assistant_tab:
    st.subheader("Study assistant")
    st.markdown(
        '<p class="section-note">Ask about any section of the proposal, methodology, Chapter Four results, DALEX analysis, or the latest prediction.</p>',
        unsafe_allow_html=True,
    )

    quick_questions = [
        "What are the study objectives?",
        "Why was XGBoost selected?",
        "What were the main DALEX findings?",
        "Explain the latest prediction.",
    ]

    quick_columns = st.columns(2)
    quick_prompt = None
    for index, question in enumerate(quick_questions):
        with quick_columns[index % 2]:
            if st.button(
                question,
                key=f"quick_{index}",
                use_container_width=True,
            ):
                quick_prompt = question

    chat_box = st.container(height=430, border=True)
    with chat_box:
        for message in st.session_state.assistant_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    typed_prompt = st.chat_input("Ask a question about the study")
    prompt = quick_prompt or typed_prompt

    if prompt:
        cleaned = prompt.strip()[:1200]
        if cleaned:
            st.session_state.assistant_messages.append(
                {"role": "user", "content": cleaned}
            )
            answer = answer_study_question(
                cleaned,
                selected_model_name,
            )
            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": "What would you like to know about the study?",
            }
        ]
        st.rerun()



# =========================================================
# METHODOLOGY
# =========================================================

with methodology_tab:
    st.subheader("Study methodology")
    st.markdown(
        '<p class="section-note">A concise overview of the analytical workflow used to develop the application.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Data and outcome", expanded=True):
        st.write(
            "The analysis used birth-history data for 5,115 children from "
            "the 2022 Kenya Demographic and Health Survey. Under-five "
            "mortality was coded as 1 for death before age five and 0 for "
            "survival."
        )

    with st.expander("Predictors and preprocessing"):
        st.write(
            "The models used maternal, child, birth, and household "
            "predictors. Missing values and extreme observations were "
            "treated, categorical variables were encoded, continuous "
            "variables were standardized where required, and class "
            "imbalance was addressed in the training data using SMOTE."
        )

    with st.expander("Model development and evaluation"):
        st.write(
            "Logistic Regression, Support Vector Machine, Random Forest, "
            "and XGBoost were developed and compared. Repeated stratified "
            "five-fold cross-validation and hyperparameter tuning were "
            "used during model development. Performance was assessed using "
            "accuracy, precision, recall, specificity, F1-score, ROC-AUC, "
            "and Cohen's Kappa."
        )

    with st.expander("Model selection and explainability"):
        st.write(
            "XGBoost was selected as the best-performing model based on its "
            "overall balance across the evaluation metrics. DALEX was used "
            "to explain global predictor importance and individual "
            "predictions through SHAP, breakdown, and partial dependence "
            "analyses."
        )

st.markdown(
    """
    <div class="footer">
        Research and educational use only. This application does not provide
        a clinical diagnosis and must not replace assessment by a qualified
        healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)
