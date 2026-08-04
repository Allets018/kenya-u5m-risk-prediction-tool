from pathlib import Path
import html
import re

import dalex as dx
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Under-Five Mortality Risk Assessment",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary:#0B6E69; --primary-dark:#07534F; --navy:#17324D;
        --text:#23313F; --muted:#667784; --surface:#FFFFFF;
        --background:#F4F8F7; --border:#D8E5E3;
        --low:#1B7F5A; --moderate:#B76E00; --high:#B42318;
    }
    .stApp {
        background:
          radial-gradient(circle at 100% 0%, #E7F3F1 0%, transparent 28%),
          linear-gradient(180deg, #F8FBFA 0%, var(--background) 100%);
        color:var(--text);
    }
    .block-container {max-width:1240px; padding-top:1.2rem; padding-bottom:3rem;}
    h1,h2,h3 {color:var(--navy); letter-spacing:-.02em;}
    h1 {font-size:clamp(2rem,4vw,3.1rem)!important; font-weight:780!important;}
    h2 {font-size:1.65rem!important; font-weight:730!important;}
    h3 {font-size:1.18rem!important; font-weight:700!important;}
    .hero {
        background:linear-gradient(120deg,rgba(11,110,105,.98),rgba(23,50,77,.96));
        border-radius:24px; padding:2rem 2.2rem; margin-bottom:1rem; color:white;
        box-shadow:0 18px 45px rgba(23,50,77,.16);
    }
    .hero-kicker {margin:0 0 .5rem; font-size:.82rem; font-weight:750;
        text-transform:uppercase; letter-spacing:.12em; opacity:.85;}
    .hero-title {margin:0; font-size:clamp(1.9rem,4vw,3rem);
        line-height:1.08; font-weight:800;}
    .hero-copy {margin:.8rem 0 0; max-width:820px; font-size:1.02rem;
        line-height:1.65; opacity:.94;}
    .section-intro {color:var(--muted); font-size:.98rem;
        margin-top:-.35rem; margin-bottom:1rem;}
    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(255,255,255,.96); border-color:var(--border)!important;
        border-radius:18px; box-shadow:0 8px 24px rgba(23,50,77,.055);
    }
    div[data-testid="stMetric"] {
        background:var(--surface); border:1px solid var(--border);
        border-radius:16px; padding:1rem 1.1rem;
        box-shadow:0 8px 24px rgba(23,50,77,.055);
    }
    div[data-baseweb="select"]>div,
    div[data-testid="stNumberInput"] input, textarea {
        background-color:#F8FBFA!important; border-color:#C9DBD8!important;
    }
    .stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {
        border-radius:11px; min-height:2.8rem; font-weight:700;
        border:1px solid transparent; transition:.18s ease;
    }
    .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {
        background:linear-gradient(135deg,var(--primary),var(--primary-dark));
        color:#fff; box-shadow:0 8px 18px rgba(11,110,105,.2);
    }
    .stButton>button:hover,.stFormSubmitButton>button:hover,
    .stDownloadButton>button:hover {transform:translateY(-1px); border-color:var(--primary);}
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#EAF5F3 0%,#F3F8F7 100%);
        border-right:1px solid #CFDFDC;
    }
    button[data-baseweb="tab"] {font-weight:700; color:#526674;}
    button[data-baseweb="tab"][aria-selected="true"] {color:var(--primary);}
    .risk-card {
        border-radius:20px; padding:1.35rem 1.5rem; background:#fff;
        border:1px solid var(--border); box-shadow:0 10px 28px rgba(23,50,77,.07);
        height:100%;
    }
    .risk-label {color:var(--muted); font-size:.83rem; font-weight:750;
        text-transform:uppercase; letter-spacing:.08em;}
    .risk-probability {color:var(--navy); font-size:2.6rem; line-height:1.1;
        font-weight:820; margin:.45rem 0;}
    .risk-badge {display:inline-block; border-radius:999px; padding:.36rem .75rem;
        font-size:.9rem; font-weight:760;}
    .risk-low {background:#E7F6EF; color:var(--low); border:1px solid #BCE4D2;}
    .risk-moderate {background:#FFF4DB; color:var(--moderate); border:1px solid #F1D396;}
    .risk-high {background:#FDECEA; color:var(--high); border:1px solid #F0C2BC;}
    .probability-track {width:100%; height:12px; background:#E7EEED;
        border-radius:999px; overflow:hidden; margin-top:1rem;}
    .probability-fill {height:100%; border-radius:999px;}
    .mini-note {margin-top:.8rem; color:var(--muted); font-size:.9rem; line-height:1.5;}
    .footer {color:#6A7C87; font-size:.82rem; text-align:center; margin-top:3rem;
        border-top:1px solid #D7E4E2; padding-top:1.2rem; line-height:1.6;}
    @media(max-width:800px){
        .hero{padding:1.5rem;border-radius:18px}
        .block-container{padding-left:1rem;padding-right:1rem}
        .risk-probability{font-size:2.1rem}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
APP_COLUMN_MAP = {model_name: app_name for app_name, model_name in MODEL_COLUMN_MAP.items()}

BACKGROUND_CATEGORY_MAP = {
    "maternal_health_status": {"Bad": "Poor", "Very bad": "Poor"},
    "multiple_birth": {"1st of multiple": "Twin birth", "2nd of multiple": "Twin birth"},
    "marital_status": {
        "Divorced": "Not married", "Living with partner": "Not married",
        "Never in union": "Not married",
        "No longer living together/separated": "Not married",
        "Widowed": "Not married",
    },
}

LABELS = {
    "maternal_age_first_birth": "Maternal age at first birth",
    "birth_interval": "Preceding birth interval",
    "anc_visits": "Antenatal care visits",
    "birth_weight": "Birth weight",
    "maternal_education": "Maternal education",
    "maternal_health_status": "Maternal health status",
    "birth_order": "Birth order",
    "wealth_index": "Household wealth index",
    "multiple_birth": "Multiple-birth status",
    "child_sex": "Child sex",
    "birth_assistance": "Birth assistance",
    "residence": "Place of residence",
    "marital_status": "Marital status",
}

FIELD_HELP = {
    "maternal_age_first_birth": "Mother's age, in completed years, at her first birth.",
    "birth_interval": "Months between the child and the immediately preceding birth.",
    "anc_visits": "Total antenatal care visits during the pregnancy.",
    "birth_weight": "Recorded birth weight in grams.",
    "maternal_education": "Highest education level attained by the mother.",
    "maternal_health_status": "Maternal health-status category in the study dataset.",
    "birth_order": "The child's position among all births to the mother.",
    "wealth_index": "Household wealth category in the study dataset.",
    "multiple_birth": "Whether the child was from a single or twin birth.",
    "child_sex": "Sex of the child.",
    "birth_assistance": "Whether birth assistance was recorded as present.",
    "residence": "Urban or rural place of residence.",
    "marital_status": "Mother's marital-status category in the analysis.",
}

FIELD_GROUPS = {
    "Maternal characteristics": [
        "maternal_age_first_birth", "maternal_education",
        "maternal_health_status", "anc_visits", "marital_status",
    ],
    "Child and birth characteristics": [
        "birth_interval", "birth_weight", "birth_order",
        "multiple_birth", "child_sex", "birth_assistance",
    ],
    "Household characteristics": ["wealth_index", "residence"],
}

CATEGORY_ORDER = {
    "maternal_education": ["No education", "Primary", "Secondary", "Higher"],
    "maternal_health_status": ["Poor", "Moderate", "Good", "Very good"],
    "birth_order": list(range(1, 15)) + [str(value) for value in range(1, 15)],
    "wealth_index": ["Poorest", "Poorer", "Middle", "Richer", "Richest"],
    "multiple_birth": ["Single birth", "Twin birth"],
    "child_sex": ["Female", "Male"],
    "birth_assistance": ["No", "Yes"],
    "residence": ["Rural", "Urban"],
    "marital_status": ["Not married", "Married"],
}

NUMERIC_CONFIG = {
    "maternal_age_first_birth": {"unit": "years", "step": 1.0, "format": "%.0f"},
    "birth_interval": {"unit": "months", "step": 1.0, "format": "%.0f"},
    "anc_visits": {"unit": "visits", "step": 1.0, "format": "%.0f"},
    "birth_weight": {"unit": "grams", "step": 10.0, "format": "%.0f"},
}


def get_preprocessor(model):
    if not hasattr(model, "named_steps"):
        raise TypeError("The saved estimator is not a fitted pipeline.")
    for step_name in ("pre_smote", "preprocessor", "preprocess"):
        if step_name in model.named_steps:
            return model.named_steps[step_name]
    raise KeyError("No fitted preprocessing step was found in the pipeline.")


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
    raise AttributeError("The categorical transformer has no fitted categories_.")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    loaded_models = {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.joblib"),
        "Support Vector Machine": joblib.load(MODEL_DIR / "svm.joblib"),
        "Random Forest": joblib.load(MODEL_DIR / "random_forest.joblib"),
        "XGBoost": joblib.load(MODEL_DIR / "xgboost.joblib"),
    }
    loaded_metadata = joblib.load(MODEL_DIR / "app_metadata.joblib")
    loaded_background = joblib.load(MODEL_DIR / "dalex_background.joblib")

    preprocessor = get_preprocessor(loaded_models["XGBoost"])
    categorical_transformer, categorical_columns = get_categorical_transformer(preprocessor)
    encoder = get_encoder(categorical_transformer)

    for model_column, categories in zip(categorical_columns, encoder.categories_):
        app_column = APP_COLUMN_MAP[model_column]
        loaded_metadata["categorical"][app_column] = [
            value.item() if isinstance(value, np.generic) else value
            for value in categories if not pd.isna(value)
        ]
    return loaded_models, loaded_metadata, loaded_background


try:
    models, metadata, background_raw = load_artifacts()
except FileNotFoundError as error:
    st.error("A required model or metadata file was not found beside app.py.")
    st.code(str(error))
    st.stop()
except Exception as error:
    st.error(
        "The saved artifacts could not be loaded. Use the same package "
        "environment used to train and save the pipelines."
    )
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")
    st.stop()


def arrange_categories(variable, available_options):
    preferred = CATEGORY_ORDER.get(variable, [])
    ordered = [value for value in preferred if value in available_options]
    remaining = [value for value in available_options if value not in ordered]
    return ordered + sorted(remaining, key=str)


def mortality_probability(model, data):
    return model.predict_proba(data)[:, 1]


def get_model_categorical_columns(model):
    _, columns = get_categorical_transformer(get_preprocessor(model))
    return columns


def prepare_model_input(values, model):
    model_data = pd.DataFrame([values]).rename(columns=MODEL_COLUMN_MAP)
    expected_columns = list(model.feature_names_in_)
    missing = [column for column in expected_columns if column not in model_data.columns]
    if missing:
        raise ValueError("Missing model columns: " + ", ".join(missing))
    model_data = model_data.reindex(columns=expected_columns)
    for column in get_model_categorical_columns(model):
        model_data[column] = model_data[column].astype(object)
    return model_data


def classify_risk(probability, configured_thresholds):
    if probability < configured_thresholds["low"]:
        return "Low risk"
    if probability < configured_thresholds["high"]:
        return "Moderate risk"
    return "High risk"


def risk_style(category):
    if category == "Low risk":
        return {"class": "risk-low", "fill": "#1B7F5A"}
    if category == "Moderate risk":
        return {"class": "risk-moderate", "fill": "#B76E00"}
    return {"class": "risk-high", "fill": "#B42318"}


def render_field(variable):
    label = LABELS.get(variable, variable)
    help_text = FIELD_HELP.get(variable)
    if variable in metadata["continuous"]:
        settings = metadata["continuous"][variable]
        config = NUMERIC_CONFIG.get(variable, {"unit": "", "step": 1.0, "format": "%.1f"})
        displayed_label = f"{label} ({config['unit']})" if config["unit"] else label
        return st.number_input(
            displayed_label,
            min_value=float(settings["min"]),
            max_value=float(settings["max"]),
            value=float(settings["default"]),
            step=float(config["step"]),
            format=config["format"],
            help=help_text,
            key=f"input_{variable}",
        )
    if variable in metadata["categorical"]:
        return st.selectbox(
            label,
            arrange_categories(variable, metadata["categorical"][variable]),
            help=help_text,
            key=f"input_{variable}",
        )
    raise KeyError(f"{variable} is missing from app metadata.")


def calculate_all_predictions(values):
    rows = []
    for model_name, model in models.items():
        model_input = prepare_model_input(values, model)
        probability = float(model.predict_proba(model_input)[0, 1])
        rows.append({
            "Model": model_name,
            "Mortality probability": probability,
            "Risk classification": classify_risk(probability, thresholds),
        })
    return pd.DataFrame(rows)


def make_prediction_report(values, comparison, selected_model_name):
    selected = comparison.loc[comparison["Model"] == selected_model_name].iloc[0]
    report = {
        "Selected model": selected_model_name,
        "Predicted mortality probability": selected["Mortality probability"],
        "Risk classification": selected["Risk classification"],
    }
    for key, value in values.items():
        report[LABELS.get(key, key)] = value
    return pd.DataFrame([report])


def extract_breakdown_summary(result, limit=5):
    if not isinstance(result, pd.DataFrame):
        return []
    contribution_col = next((c for c in ("contribution", "_contribution_") if c in result.columns), None)
    variable_col = next((c for c in ("variable", "variable_name", "_variable_") if c in result.columns), None)
    if contribution_col is None or variable_col is None:
        return []
    data = result.copy()
    data = data[~data[variable_col].astype(str).str.startswith("_")]
    data["abs_contribution"] = data[contribution_col].abs()
    data = data.sort_values("abs_contribution", ascending=False).head(limit)
    return [
        {"variable": str(row[variable_col]), "contribution": float(row[contribution_col])}
        for _, row in data.iterrows()
    ]


background = background_raw.copy()
for variable, replacements in BACKGROUND_CATEGORY_MAP.items():
    if variable in background.columns:
        background[variable] = background[variable].replace(replacements)
background = background.rename(columns=MODEL_COLUMN_MAP)
reference_model = models["XGBoost"]
background = background.reindex(columns=reference_model.feature_names_in_)
for variable in get_model_categorical_columns(reference_model):
    background[variable] = background[variable].astype(object).where(
        background[variable].notna(), np.nan
    )


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


thresholds = metadata.get("thresholds", {"low": 0.20, "high": 0.50})
thresholds = {"low": float(thresholds["low"]), "high": float(thresholds["high"])}
if thresholds["low"] >= thresholds["high"]:
    st.error("The low-risk threshold must be smaller than the high-risk threshold.")
    st.stop()

st.session_state.setdefault("last_prediction_values", None)
st.session_state.setdefault("assistant_messages", [{
    "role": "assistant",
    "content": (
        "Hello. I can explain the study, model metrics, predictors, risk "
        "categories and DALEX results. I can also interpret the latest "
        "prediction without providing a clinical diagnosis."
    ),
}])
st.session_state.setdefault("breakdown_summary", [])
st.session_state.setdefault("breakdown_figure", None)
st.session_state.setdefault("breakdown_signature", None)


def clear_application():
    for key in list(st.session_state.keys()):
        if key.startswith("input_"):
            del st.session_state[key]
    st.session_state.last_prediction_values = None
    st.session_state.breakdown_summary = []
    st.session_state.breakdown_figure = None
    st.session_state.breakdown_signature = None


# =========================================================
# SIDEBAR
# =========================================================

model_names = list(models.keys())
with st.sidebar:
    st.markdown("### Model settings")
    selected_model_name = st.selectbox(
        "Prediction model",
        model_names,
        index=model_names.index("XGBoost"),
        help=(
            "XGBoost is the study-selected best-performing model. "
            "The other models remain available for comparison."
        ),
    )

    if selected_model_name == "XGBoost":
        st.success("Study-selected best-performing model")
    else:
        st.info("Comparison model selected. XGBoost remains the final study model.")

    st.markdown("---")
    st.markdown("### Study-specific risk bands")
    st.write(f"**Low:** below {thresholds['low']:.0%}")
    st.write(
        f"**Moderate:** {thresholds['low']:.0%} to "
        f"below {thresholds['high']:.0%}"
    )
    st.write(f"**High:** {thresholds['high']:.0%} or above")
    st.caption(
        "These are research classifications and have not been clinically validated."
    )

    st.markdown("---")
    if st.button("Reset assessment", use_container_width=True):
        clear_application()
        st.rerun()

    st.caption(
        "Avoid entering names, identification numbers, addresses "
        "or medical-record identifiers."
    )


# =========================================================
# HEADER AND NAVIGATION
# =========================================================

st.markdown(
    """
    <section class="hero">
        <p class="hero-kicker">Explainable machine-learning research prototype</p>
        <p class="hero-title">Under-Five Mortality Risk Assessment</p>
        <p class="hero-copy">
            Enter maternal, child and household characteristics to obtain a
            study-specific mortality probability, compare four models and
            review an individual DALEX explanation.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research and educational use only. This application does not provide "
    "a clinical diagnosis and must not replace assessment by a qualified "
    "healthcare professional.",
    icon="⚠️",
)

assessment_tab, explanation_tab, assistant_tab, about_tab = st.tabs(
    [
        "Risk assessment",
        "Model comparison and explanation",
        "Study assistant",
        "About the tool",
    ]
)


# =========================================================
# RISK ASSESSMENT
# =========================================================

with assessment_tab:
    st.subheader("Enter the child profile")
    st.markdown(
        '<p class="section-intro">'
        "Complete all fields below. The saved pipeline applies the same "
        "transformations used during model development."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("risk_assessment_form"):
        user_values = {}

        for group_name, variables in FIELD_GROUPS.items():
            with st.container(border=True):
                st.markdown(f"### {group_name}")
                columns = st.columns(2)
                for index, variable in enumerate(variables):
                    with columns[index % 2]:
                        user_values[variable] = render_field(variable)

        submitted = st.form_submit_button(
            "Generate risk assessment",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        st.session_state.last_prediction_values = user_values.copy()
        st.session_state.breakdown_summary = []
        st.session_state.breakdown_figure = None
        st.session_state.breakdown_signature = None

    values_for_prediction = st.session_state.last_prediction_values

    if values_for_prediction is None:
        st.info(
            "Complete the form and select **Generate risk assessment** "
            "to view the result."
        )
    else:
        try:
            comparison = calculate_all_predictions(values_for_prediction)
            selected_row = comparison.loc[
                comparison["Model"] == selected_model_name
            ].iloc[0]
            selected_probability = float(selected_row["Mortality probability"])
            risk_category = selected_row["Risk classification"]
            style = risk_style(risk_category)
        except Exception as error:
            st.error(
                "The prediction could not be completed. Confirm that the "
                "saved models, metadata and categories came from the same pipeline."
            )
            with st.expander("Technical details"):
                st.code(f"{type(error).__name__}: {error}")
            st.stop()

        st.subheader("Assessment result")
        result_left, result_right = st.columns([1.25, 1])

        with result_left:
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-label">Predicted probability of under-five death</div>
                    <div class="risk-probability">{selected_probability:.1%}</div>
                    <span class="risk-badge {style['class']}">
                        {html.escape(risk_category)}
                    </span>
                    <div class="probability-track">
                        <div class="probability-fill"
                             style="width:{selected_probability * 100:.1f}%;
                                    background:{style['fill']};">
                        </div>
                    </div>
                    <div class="mini-note">
                        Prediction generated using
                        <strong>{html.escape(selected_model_name)}</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_right:
            with st.container(border=True):
                st.markdown("### How to read this result")
                if risk_category == "Low risk":
                    st.success(
                        "The profile falls within the study's low-risk probability band."
                    )
                elif risk_category == "Moderate risk":
                    st.warning(
                        "The profile falls within the study's moderate-risk probability band."
                    )
                else:
                    st.error(
                        "The profile falls within the study's high-risk probability band."
                    )

                st.write(
                    "The probability is a model estimate for the entered profile. "
                    "It does not guarantee an individual outcome."
                )

        report = make_prediction_report(
            values_for_prediction,
            comparison,
            selected_model_name,
        )

        action_columns = st.columns(2)
        with action_columns[0]:
            st.download_button(
                "Download prediction summary",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="under_five_risk_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with action_columns[1]:
            with st.expander("Review submitted profile"):
                profile_display = pd.DataFrame({
                    "Variable": [
                        LABELS.get(key, key)
                        for key in values_for_prediction
                    ],
                    "Entered value": list(values_for_prediction.values()),
                })
                st.dataframe(
                    profile_display,
                    hide_index=True,
                    use_container_width=True,
                )


# =========================================================
# MODEL COMPARISON AND DALEX
# =========================================================

with explanation_tab:
    values_for_prediction = st.session_state.last_prediction_values

    if values_for_prediction is None:
        st.info(
            "Generate a risk assessment first. The model comparison and "
            "DALEX explanation will then appear here."
        )
    else:
        comparison = calculate_all_predictions(values_for_prediction)
        comparison["Selected"] = np.where(
            comparison["Model"] == selected_model_name,
            "Selected",
            "",
        )

        st.subheader("Comparison across the four models")
        st.markdown(
            '<p class="section-intro">'
            "Different algorithms may assign different probabilities to "
            "the same profile. The active model is identified in the table."
            "</p>",
            unsafe_allow_html=True,
        )

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
                    "Risk classification",
                    "Selected",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        chart_data = comparison.sort_values(
            "Mortality probability",
            ascending=True,
        )

        comparison_figure = px.bar(
            chart_data,
            x="Mortality probability",
            y="Model",
            orientation="h",
            color="Risk classification",
            text="Mortality probability",
            color_discrete_map={
                "Low risk": "#1B7F5A",
                "Moderate risk": "#B76E00",
                "High risk": "#B42318",
            },
            category_orders={
                "Risk classification": [
                    "Low risk",
                    "Moderate risk",
                    "High risk",
                ]
            },
        )

        comparison_figure.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
        )
        comparison_figure.update_layout(
            height=390,
            xaxis_title="Predicted mortality probability",
            yaxis_title="",
            xaxis_tickformat=".0%",
            xaxis_range=[0, 1],
            legend_title="Risk classification",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#23313F"),
            margin=dict(l=20, r=55, t=20, b=20),
        )

        st.plotly_chart(comparison_figure, use_container_width=True)

        st.subheader("Individual DALEX breakdown")
        st.markdown(
            '<p class="section-intro">'
            "The breakdown starts from the model average and shows which "
            "entered values increased or decreased this prediction."
            "</p>",
            unsafe_allow_html=True,
        )

        input_data = prepare_model_input(
            values_for_prediction,
            models[selected_model_name],
        )
        signature = (
            selected_model_name,
            tuple((key, str(value)) for key, value in values_for_prediction.items()),
        )

        if (
            st.session_state.breakdown_signature is not None
            and st.session_state.breakdown_signature != signature
        ):
            st.session_state.breakdown_figure = None
            st.session_state.breakdown_summary = []
            st.session_state.breakdown_signature = None

        if st.button(
            "Generate DALEX explanation",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Calculating predictor contributions..."):
                    explainer = build_explainer(selected_model_name)
                    breakdown = explainer.predict_parts(
                        new_observation=input_data,
                        type="break_down",
                    )
                    st.session_state.breakdown_figure = breakdown.plot(
                        max_vars=10,
                        show=False,
                    )
                    st.session_state.breakdown_summary = extract_breakdown_summary(
                        breakdown.result,
                        limit=5,
                    )
                    st.session_state.breakdown_signature = signature
            except Exception as error:
                st.warning(
                    "The prediction was completed, but the DALEX explanation "
                    "could not be generated."
                )
                with st.expander("Technical details"):
                    st.code(f"{type(error).__name__}: {error}")

        if st.session_state.breakdown_figure is not None:
            st.plotly_chart(
                st.session_state.breakdown_figure,
                use_container_width=True,
            )
            st.caption(
                "Positive contributions increase predicted mortality risk, "
                "while negative contributions decrease it. These are model "
                "explanations and do not establish causality."
            )


# =========================================================
# STUDY ASSISTANT
# =========================================================

def build_study_knowledge_base():
    return [
        {
            "topic": "Study purpose",
            "keywords": "purpose aim objective study research Kenya",
            "answer": (
                "The study develops and evaluates machine-learning models "
                "for predicting under-five mortality in Kenya and applies "
                "DALEX to explain the selected model."
            ),
        },
        {
            "topic": "Outcome",
            "keywords": "outcome target death survival class one zero",
            "answer": (
                "The outcome is under-five mortality. Class 1 represents "
                "death before age five, while class 0 represents survival."
            ),
        },
        {
            "topic": "Predictors",
            "keywords": "predictor feature variable maternal child household input",
            "answer": (
                "The tool uses 13 predictors: maternal age at first birth, "
                "preceding birth interval, antenatal care visits, birth weight, "
                "maternal education, maternal health status, birth order, "
                "household wealth, multiple-birth status, child sex, birth "
                "assistance, residence and marital status."
            ),
        },
        {
            "topic": "Recall",
            "keywords": "recall sensitivity false negative missed death",
            "answer": (
                "Recall is the proportion of actual under-five deaths correctly "
                "identified. A false negative is a death predicted as survival."
            ),
        },
        {
            "topic": "Precision",
            "keywords": "precision positive predictive false positive",
            "answer": (
                "Precision is the proportion of predicted deaths that are actual "
                "deaths. Lower precision means more surviving children are "
                "incorrectly flagged as deaths."
            ),
        },
        {
            "topic": "Specificity",
            "keywords": "specificity true negative survivor correctly identified",
            "answer": (
                "Specificity is the proportion of actual survivors correctly "
                "classified as survival."
            ),
        },
        {
            "topic": "Accuracy and imbalance",
            "keywords": "accuracy imbalance class imbalance misleading",
            "answer": (
                "Accuracy is the overall proportion classified correctly. "
                "Because deaths are the minority class, accuracy should be "
                "interpreted with recall, precision, specificity, F1-score, "
                "ROC-AUC and Cohen's Kappa."
            ),
        },
        {
            "topic": "F1-score",
            "keywords": "f1 harmonic precision recall",
            "answer": (
                "The F1-score is the harmonic mean of precision and recall. "
                "It summarizes the balance between detecting deaths and "
                "limiting false alarms."
            ),
        },
        {
            "topic": "ROC-AUC",
            "keywords": "roc auc receiver operating discrimination curve",
            "answer": (
                "ROC-AUC measures how well a model ranks deaths above survivors "
                "across possible classification thresholds. Values closer to 1 "
                "indicate stronger discrimination."
            ),
        },
        {
            "topic": "Cohen's Kappa",
            "keywords": "cohen kappa agreement chance",
            "answer": (
                "Cohen's Kappa measures agreement between observed and predicted "
                "outcomes after accounting for agreement expected by chance."
            ),
        },
        {
            "topic": "SMOTENC",
            "keywords": "smote smotenc oversampling imbalance synthetic",
            "answer": (
                "SMOTENC was used to address class imbalance in training data "
                "containing numeric and categorical predictors. It was not "
                "applied to the test set."
            ),
        },
        {
            "topic": "Cross-validation",
            "keywords": (
                "cross validation repeated stratified five fold grid search "
                "tuning hyperparameter"
            ),
            "answer": (
                "Repeated stratified five-fold cross-validation was used to "
                "obtain stable training estimates while preserving class "
                "proportions. Grid search compared candidate settings."
            ),
        },
        {
            "topic": "Best model",
            "keywords": "best model xgboost selected model overall performance",
            "answer": (
                "XGBoost was selected because it provided the strongest overall "
                "balance across accuracy, precision, recall, specificity, "
                "F1-score, ROC-AUC and Cohen's Kappa."
            ),
        },
        {
            "topic": "DALEX",
            "keywords": "dalex explainable ai interpretation explanation",
            "answer": (
                "DALEX is the explainability framework used in the study. It "
                "provides global variable importance, partial dependence, SHAP "
                "and individual breakdown explanations."
            ),
        },
        {
            "topic": "Breakdown plot",
            "keywords": (
                "breakdown contribution individual explanation increase decrease"
            ),
            "answer": (
                "A breakdown plot starts from the model's average prediction "
                "and adds each predictor's contribution for one profile. "
                "Positive contributions increase risk and negative contributions "
                "decrease it."
            ),
        },
        {
            "topic": "Risk categories",
            "keywords": "risk probability low moderate high threshold category",
            "answer": (
                f"The study-specific categories are low risk below "
                f"{thresholds['low']:.0%}, moderate risk from "
                f"{thresholds['low']:.0%} to below {thresholds['high']:.0%}, "
                f"and high risk at {thresholds['high']:.0%} or above. They are "
                "not clinically validated cut-offs."
            ),
        },
        {
            "topic": "Limitations",
            "keywords": "clinical diagnosis doctor treatment limitation validated",
            "answer": (
                "The application is a research and educational prototype. It "
                "must not replace clinical assessment, diagnosis or treatment."
            ),
        },
        {
            "topic": "Privacy",
            "keywords": "privacy stored personal data name identification",
            "answer": (
                "The app does not intentionally store submitted profiles. Do "
                "not enter names, identification numbers, addresses or "
                "medical-record identifiers."
            ),
        },
    ]


def latest_prediction_context():
    values = st.session_state.last_prediction_values
    if values is None:
        return None
    comparison = calculate_all_predictions(values)
    row = comparison.loc[comparison["Model"] == selected_model_name].iloc[0]
    return {
        "model": selected_model_name,
        "probability": float(row["Mortality probability"]),
        "risk": row["Risk classification"],
        "contributions": st.session_state.breakdown_summary,
    }


def answer_study_question(question):
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = " ".join(normalized.split())

    emergency_phrases = (
        "not breathing", "unconscious", "seizure", "severe bleeding",
        "child is dying", "medical emergency",
    )
    if any(phrase in normalized for phrase in emergency_phrases):
        return (
            "This research assistant cannot assess an emergency. Seek "
            "immediate help from an appropriate local emergency medical "
            "service or qualified healthcare professional."
        )

    context = latest_prediction_context()
    interpretation_phrases = (
        "this result", "my result", "latest prediction",
        "what does the prediction mean", "why high risk",
        "why moderate risk", "why low risk", "explain the prediction",
    )

    if context is not None and any(
        phrase in normalized for phrase in interpretation_phrases
    ):
        response = (
            f"The latest assessment used {context['model']} and produced a "
            f"predicted mortality probability of {context['probability']:.1%}, "
            f"classified as {context['risk']}. This is a study-specific "
            f"estimate, not a clinical diagnosis."
        )
        contributions = context["contributions"]
        if contributions:
            increasing = [x for x in contributions if x["contribution"] > 0][:3]
            decreasing = [x for x in contributions if x["contribution"] < 0][:3]
            if increasing:
                response += (
                    " The strongest recorded contributors increasing the "
                    "prediction were "
                    + ", ".join(item["variable"] for item in increasing)
                    + "."
                )
            if decreasing:
                response += (
                    " Contributors decreasing the prediction included "
                    + ", ".join(item["variable"] for item in decreasing)
                    + "."
                )
        else:
            response += (
                " Generate the DALEX explanation in the comparison tab to "
                "identify the main increasing and decreasing contributions."
            )
        return response

    knowledge_base = build_study_knowledge_base()
    documents = [
        " ".join([entry["topic"], entry["keywords"], entry["answer"]])
        for entry in knowledge_base
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents + [normalized])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    if best_score < 0.07:
        return (
            "I could not confidently match that question to the study "
            "knowledge base. Ask about the study purpose, predictors, metrics, "
            "SMOTENC, cross-validation, XGBoost, risk thresholds or DALEX."
        )
    return knowledge_base[best_index]["answer"]


with assistant_tab:
    st.subheader("Study knowledge assistant")
    st.markdown(
        '<p class="section-intro">'
        "Ask study-specific questions or request an interpretation of the "
        "latest prediction. This assistant uses a curated local knowledge "
        "base and does not send questions to an external AI service."
        "</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "Do not enter names, identification numbers, addresses or "
        "medical-record information.",
        icon="🔒",
    )

    quick_prompts = [
        "Why was XGBoost selected?",
        "What does recall mean?",
        "How does DALEX explain a prediction?",
        "Explain the latest prediction.",
    ]
    prompt_columns = st.columns(2)
    selected_quick_prompt = None

    for index, prompt in enumerate(quick_prompts):
        with prompt_columns[index % 2]:
            if st.button(
                prompt,
                key=f"quick_prompt_{index}",
                use_container_width=True,
            ):
                selected_quick_prompt = prompt

    chat_container = st.container(height=430, border=True)
    with chat_container:
        for message in st.session_state.assistant_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    typed_prompt = st.chat_input(
        "Ask a question about the study or latest prediction"
    )
    active_prompt = selected_quick_prompt or typed_prompt

    if active_prompt:
        cleaned_prompt = active_prompt.strip()[:1000]
        if cleaned_prompt:
            st.session_state.assistant_messages.append(
                {"role": "user", "content": cleaned_prompt}
            )
            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": answer_study_question(cleaned_prompt)}
            )
            st.rerun()

    if st.button("Clear assistant conversation", use_container_width=True):
        st.session_state.assistant_messages = [{
            "role": "assistant",
            "content": (
                "The conversation has been cleared. What would you like "
                "to know about the study?"
            ),
        }]
        st.rerun()


# =========================================================
# ABOUT
# =========================================================

with about_tab:
    st.subheader("About this research prototype")
    about_left, about_right = st.columns([1.25, 1])

    with about_left:
        with st.container(border=True):
            st.markdown("### Purpose")
            st.write(
                "The application estimates under-five mortality probability "
                "from maternal, child and household characteristics using "
                "saved machine-learning pipelines."
            )
            st.markdown("### Models")
            st.write(
                "Logistic Regression, Support Vector Machine, Random Forest "
                "and XGBoost are available. XGBoost is the study-selected "
                "best-performing model."
            )
            st.markdown("### Explainability")
            st.write(
                "DALEX shows which predictors increased or decreased the "
                "selected model's prediction for the submitted profile."
            )

    with about_right:
        with st.container(border=True):
            st.markdown("### Analysis workflow")
            st.write("1. Validate the entered profile")
            st.write("2. Apply the saved preprocessing pipeline")
            st.write("3. Generate probabilities from all four models")
            st.write("4. Assign the study-specific risk category")
            st.write("5. Explain the selected model using DALEX")

    st.warning(
        "The application has not undergone external clinical validation. "
        "Its outputs are research predictions rather than clinical decisions."
    )


st.markdown(
    """
    <div class="footer">
        Under-Five Mortality Risk Assessment Tool — Research Prototype
        <br>
        Submitted profiles are not intentionally stored. Avoid entering
        personally identifiable or medical-record information.
    </div>
    """,
    unsafe_allow_html=True,
)
