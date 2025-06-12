import subprocess
import sys
import time
import signal

"""
This script selects and runs an emotion prediction model pipeline based on
the specified model type.

- Baseline: Runs BaselineModel.pipeline() and prints the result.
- Bert: Runs BertModel.pipeline().
- Bert_p: Interactively prompts user for predictions
using PredictEmotion.predict().
"""


def start_uvicorn() -> subprocess.Popen:
    """
    Launch a Uvicorn server for the FastAPI deployment.

    Returns:
        subprocess.Popen: The process handle for the Uvicorn server.
    """
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "tweet_sentiment_analysis.deployment.deploy_model:app",
        "--reload",
        "--host", "127.0.0.1",
        "--port", "8000"
    ])


def start_streamlit() -> subprocess.Popen:
    """
    Launch the Streamlit demo application.

    Returns:
        subprocess.Popen: The process handle for the Streamlit app.
    """
    return subprocess.Popen([
        "streamlit", "run",
        "tweet_sentiment_analysis/demo.py"
    ])


if __name__ == "__main__":
    procs = []
    try:
        p1 = start_uvicorn()
        procs.append(p1)

        time.sleep(3)

        p2 = start_streamlit()
        procs.append(p2)

        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
