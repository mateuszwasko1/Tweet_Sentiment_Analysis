import subprocess
import sys
import time
import signal

def start_uvicorn():
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "project_name.deployment.deploy_model:app",
        "--reload",
        "--host", "127.0.0.1",
        "--port", "8000"
    ])

def start_streamlit():
    return subprocess.Popen([
        "streamlit", "run",
        "project_name/demo.py"
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