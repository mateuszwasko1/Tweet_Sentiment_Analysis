import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Tweet Emotion Classifier",
    layout="centered",
)
st.markdown(
    """
    <style>
      .stButton>button {
          background-color: #1f77b4;
          color: white;
          border-radius: 5px;
          font-weight: bold;
      }
      .stButton>button:hover {
          background-color: #155d8b;
      }
      .stTable thead th {
          background-color: #4CAF50;
          color: white;
          text-transform: uppercase;
      }
      .main-title {
          font-size: 2.5rem;
          color: #d62728;
          font-weight: bold;
      }
      .info-panel {
          background-color: #f0f8ff;
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 20px;
      }
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
      Click <strong>Classify</strong> to send them to the FastAPI server’s 
      <code>/predict</code> endpoint and see the predicted emotions.
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
        st.error("❗ Please enter at least one tweet.")
    else:
        payload = [{"text": text} for text in lines]

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error querying FastAPI: {e}")
        else:
            try:
                results = response.json()
                df_results = pd.DataFrame(results)
            except ValueError:
                st.error("FastAPI returned invalid JSON.")
            else:
                st.success(f"Classified {len(results)} tweet{'s' if len(results)>1 else ''}.")
                st.table(df_results)

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
