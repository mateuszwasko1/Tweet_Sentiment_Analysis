import requests
import pandas as pd
import streamlit as st
import time

DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 3
INITIAL_TIMEOUT = 30  # seconds
TIMEOUT_MULTIPLIER = 2

st.set_page_config(page_title="Tweet Emotion Classifier", layout="centered")
model_choice = st.sidebar.radio(
    "Select model:",
    ["RoBERTa", "Baseline Logistic Regression"]
)
BASE_URL = "http://localhost:8000"
INFO_URL = f"{BASE_URL}/info?baseline={int(model_choice != 'RoBERTa')}"
API_URL = f"{BASE_URL}/predict?baseline={int(model_choice != 'RoBERTa')}"
batch_size = st.sidebar.slider(
    "Batch size (tweets/request)", 1, 200, DEFAULT_BATCH_SIZE, 1
)

st.markdown("""
<style>
  .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; font-weight: bold; }
  .stButton>button:hover { background-color: #155d8b; }
  .stTable thead th { background-color: #4CAF50; color: white; text-transform: uppercase; }
  .main-title { font-size: 2.5rem; color: #d62728; font-weight: bold; }
  .info-panel { background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
  .footer { border-top: 2px solid #2ca02c; padding-top: 10px; margin-top: 20px; color: #2ca02c; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Tweet Emotion Classifier Demo</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="info-panel">
  Using <strong>{model_choice}</strong>. Upload a CSV or paste tweets below. The app dynamically adjusts batch size and timeout on timeouts.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
texts = []
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(df.head())
        text_cols = df.select_dtypes(include=[object]).columns.tolist()
        if text_cols:
            col = st.selectbox("Select text column", text_cols)
            texts = df[col].astype(str).dropna().tolist()
        else:
            st.error("No text columns detected in CSV.")
    except Exception as e:
        st.error(f"CSV read error: {e}")
if not texts:
    area = st.text_area("Enter tweets (one per line)", height=150)
    if area:
        texts = [l.strip() for l in area.splitlines() if l.strip()]
if texts:
    st.info(f"Loaded {len(texts)} tweets.")

if st.button("Classify"):
    if not texts:
        st.error("Please provide tweets via CSV or text area.")
    else:
        bs = batch_size
        timeout = INITIAL_TIMEOUT
        while bs >= 1:
            try:
                total = len(texts)
                total_batches = (total + bs - 1) // bs
                results = []
                progress = st.progress(0)
                for idx in range(total_batches):
                    batch = [{"text": t} for t in texts[idx*bs:(idx+1)*bs]]
                    for attempt in range(1, MAX_RETRIES+1):
                        try:
                            resp = requests.post(API_URL, json=batch, timeout=timeout)
                            resp.raise_for_status()
                            results.extend(resp.json())
                            break
                        except requests.exceptions.Timeout:
                            if attempt == MAX_RETRIES:
                                raise
                            time.sleep(1)
                        except Exception as e:
                            raise Exception(f"Batch {idx+1}/{total_batches} failed: {e}")
                    progress.progress((idx+1)/total_batches)

                df_res = pd.DataFrame(results)
                st.success(f"Classified {len(results)} tweets with batch size {bs} and timeout {timeout}s using {model_choice}.")
                st.subheader("Full Results")
                st.dataframe(df_res)

                csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv_bytes,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

                if 'prediction' in df_res.columns:
                    summary = df_res['prediction'].value_counts().rename_axis('emotion').reset_index(name='count')
                    st.subheader("Emotion Summary")
                    st.bar_chart(summary.set_index('emotion')['count'])
                break

            except Exception as e:
                if "timed out" in str(e).lower() and bs > 1:
                    st.warning(
                        f"Timeout with batch size {bs} and timeout {timeout}s. Halving batch size and doubling timeout."
                    )
                    bs //= 2
                    timeout *= TIMEOUT_MULTIPLIER
                    continue
                st.error(f"Error: {e}")
                st.stop()

try:
    info_resp = requests.get(INFO_URL, timeout=5)
    info_resp.raise_for_status()
    info = info_resp.json()
except Exception:
    info = {"model": "Unknown", "classes": []}

model_name = info.get('model', 'Unknown')
classes = info.get('classes', [])
st.markdown(f"""
<div class="footer">
  <strong>Model:</strong> {model_name}<br>
  <strong>Classes:</strong> {', '.join(classes)}
</div>
""", unsafe_allow_html=True)