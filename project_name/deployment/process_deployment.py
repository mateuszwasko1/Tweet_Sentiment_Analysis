import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))

app = FastAPI()


def predict_emotion(text: str) -> str:
    model_loader = ModelSaver()
    model = model_loader.load_model("baseline_model")
    preprocessor = BaselinePreprocessor()
    tweet = pd.DataFrame({"tweet": [text]})
    tweet_cleaned = preprocessor.preprocessing_pipeline(at_inference=True,
                                                        data=tweet)
    prediction = model.predict(tweet_cleaned)
    if isinstance(prediction, (list, tuple, np.ndarray)):
        return prediction[0]
    return prediction
