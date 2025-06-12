from project_name.models.bert_model import BertModel
from project_name.models.baseline import BaselineModel
from project_name.deployment.process_deployment import PredictEmotion

"""if __name__ == '__main__':
    type_of_model = "Baseline"
    if type_of_model == "Baseline":
        baseline = BaselineModel()
        print(baseline.pipeline())
    elif type_of_model == "Bert":
        model = BertModel()
        model.pipeline()
    elif type_of_model == "Bert_p":
        prediction = PredictEmotion()
        number_of_predictions = int(input("How many predictions would you like\
        to make?"))
        if number_of_predictions <= 0:
            raise ValueError("Number of predictions must be greater than 0.")
        i = 0
        while i < number_of_predictions:
            i += 1
            text = input("What text would you like predict?")
            label_class, prob = prediction.predict(text)
            print(f"The predicted class is {label_class} with a probability of\
                  {(prob*100):.2f}%.")
            # print(prediction.predict(text))
"""
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
        # 1) launch uvicorn
        p1 = start_uvicorn()
        procs.append(p1)

        time.sleep(3)

        # 2) launch streamlit
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