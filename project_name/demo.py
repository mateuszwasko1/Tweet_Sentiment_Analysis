
"""
Streamlit application for interactive Tweet Emotion Classification.
Provides a UI to input tweets, send them to a FastAPI backend,
and display results.
"""
import requests
import pandas as pd
import streamlit as st

API_URL: str = "http://localhost:8000/predict"

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

st.markdown(
    '<div class="main-title">Tweet Emotion Classifier Demo</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="info-panel">
      Paste one or more tweets below (one per line).<br>
      Click <strong>Classify</strong> to send them to the FastAPI server’s
      <code>/predict</code> endpoint and see the predicted emotions.
    </div>
    """,
    unsafe_allow_html=True,
)

tweets_input: str = st.text_area(
    label="Enter tweets (one per line):",
    height=150,
    placeholder="I love you\nI hate Mondays",
)

if st.button("Classify"):
    lines: list[str] = [
        line.strip() for line in tweets_input.splitlines() if line.strip()
    ]
    if not lines:
        st.error("Please enter at least one tweet.")
    else:
        payload: list[dict[str, str]] = [{"text": text} for text in lines]

        try:
            response: requests.Response = requests.post(
                API_URL, json=payload, timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error querying FastAPI: {e}")
        else:
            try:
                results: list[dict[str, str]] = response.json()
                df_results: pd.DataFrame = pd.DataFrame(results)
            except ValueError:
                st.error("FastAPI returned invalid JSON.")
            else:
                st.success(
                    f"Classified {
                        len(results)} tweet{'s' if len(results) > 1 else ''
                                            }."
                )
                st.table(df_results)


@st.cache_data(show_spinner=False)
def get_model_info() -> dict[str, object]:
    """
    Retrieve model metadata from the FastAPI backend.

    Returns:
        A dict containing 'model' and 'classes'.
    """
    try:
        info_response: requests.Response = requests.get(
            "http://localhost:8000/info", timeout=5
        )
        info_response.raise_for_status()
        return info_response.json()
    except Exception:
        return {"model": "Unknown", "classes": []}


model_info: dict[str, object] = get_model_info()
model_name: str = model_info.get("model", "Unknown")
classes: str = ", ".join(model_info.get("classes", []))

st.markdown(
    f"""
    <div class="footer">
      <hr>
      <strong>Model:</strong> {model_name}<br>
      <strong>Classes:</strong> {classes}
    </div>
    """,
    unsafe_allow_html=True,
)
