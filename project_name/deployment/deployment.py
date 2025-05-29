import os
from fastapi import FastAPI
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import BaselinePreprocessor

app = FastAPI()


# if __name__ == "__main__":
def predict_emotion(text: str) -> str:
    model_loader = ModelSaver()
    model = model_loader.load_model("baseline_model")
    preprocessor = BaselinePreprocessor()
    # string = "I love you"
    tweet = pd.DataFrame({"tweet": [text]})
    tweet_cleaned = preprocessor.preprocessing_pipeline(at_inference=True, data=tweet)
    prediction = model.predict(tweet_cleaned)
    return (f"Prediction for '{text}': {prediction}")
    
