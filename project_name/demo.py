import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
import streamlit as st
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import BaselinePreprocessor

st.set_page_config(
    page_title="Tweet Emotion Classifier",
    layout="centered"
)

@st.cache_resource
def load_model_and_preprocessor():
    """
    Load your saved Logistic Regression model (via ModelSaver) and the
    BaselinePreprocessor. Because of @st.cache_resource, this runs only once
    per Streamlit session, then caches the result.
    """
    model_loader = ModelSaver()
    model = model_loader.load_model("baseline_model")
    preprocessor = BaselinePreprocessor()
    return model, preprocessor

st.markdown(
    """
    <style>
    /* Change the "Classify" button color */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Hover effect for button */
    .stButton>button:hover {
        background-color: #155d8b;
    }
    /* Style the table header */
    .stTable thead th {
        background-color: #ffdd57;
        color: black;
    }
    /* Color for the main title */
    .main-title {
        font-size: 2.5rem;
        color: #d62728;
        font-weight: bold;
    }
    /* Light-blue panel for instructions */
    .info-panel {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    /* Colored footer separator */
    .footer {
        border-top: 2px solid #2ca02c;
        padding-top: 10px;
        margin-top: 20px;
        color: #2ca02c;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Tweet Emotion Classifier Demo</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-panel">
    Paste one or more tweets below (one per line).<br>
    When you click <strong>Classify</strong>, each line is passed through our 
    Logistic Regression–based emotion classifier.
    </div>
    """,
    unsafe_allow_html=True
)

tweets_input = st.text_area(
    label="Enter tweets (one per line):",
    height=150,
    placeholder="I love you\nI hate Mondays"
)

if st.button("Classify"):
    lines = [line.strip() for line in tweets_input.splitlines() if line.strip()]
    if not lines:
        st.error("Please enter at least one tweet.")
    else:
        model, preprocessor = load_model_and_preprocessor()

        df_input = pd.DataFrame({"tweet": lines})
        df_clean = preprocessor.preprocessing_pipeline(at_inference=True, data=df_input)
        preds = model.predict(df_clean)

        if isinstance(preds, (list, tuple, np.ndarray)):
            preds_list = [str(p) for p in preds]
        else:
            preds_list = [str(preds)]

        results_df = pd.DataFrame({
            "Input Tweet": lines,
            "Predicted Emotion": preds_list
        })

        st.success(f"Classified {len(lines)} tweet{'s' if len(lines) > 1 else ''}.")
        st.table(results_df)

st.markdown(
    """
    <div class="footer">
    <hr>
    <strong>Model:</strong> Logistic Regression<br>
    <strong>Classes:</strong> anger, joy, sadness, fear
    </div>
    """,
    unsafe_allow_html=True
)
