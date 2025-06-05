import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
from project_name.preprocessing.ekphrasis_preprocessing import (
    MainPreprocessing)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import torch.nn.functional as F
from fastapi import FastAPI

app = FastAPI()


class PredictEmotion():
    def __init__(self, baseline=False):
        self.baseline = baseline
        if baseline:
            model_loader = ModelSaver()
            self.model = model_loader.load_model("baseline_model")
            self.preprocessor = BaselinePreprocessor()
        else:  # Use BERT model
            bert_model_path = "models/saved_bert/model"
            bert_label_encoder_path = "models/saved_bert/" \
                                      "label_encoder"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                bert_model_path)
            self.label_encoder = joblib.load(bert_label_encoder_path)
            self.preprocessor = MainPreprocessing()

    def predict(self, text):
        if self.baseline:
            prediction = self.model.predict(text)
            probability = self.model.predict_proba(text)
            confidence = np.max(probability, axis=1)[0]
            return prediction, confidence
        else:
            train_encodings = self.bert_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt")

            with torch.no_grad():
                logits = self.model(**train_encodings).logits
                probability = F.softmax(logits, dim=1)

                prob_val, predicted_class = torch.max(probability, dim=1)
                predicted_label = self.label_encoder.inverse_transform(
                    [predicted_class.item()])[0]
                confidence = prob_val.item()
            return str(predicted_label), confidence

    def output_emotion(self, text: str) -> str:
        tweet = pd.DataFrame({"tweet": [text]})
        tweet_cleaned = self.preprocessor.preprocessing_pipeline(
            at_inference=True, data=tweet)
        prediction, confidence = self.predict(tweet_cleaned)
        if isinstance(prediction, (np.ndarray, list)):
            return str(prediction[0]), float(round(confidence, 2))
        return str(prediction), float(round(confidence, 2))


if __name__ == "__main__":
    predictor = PredictEmotion(baseline=True)
    text = "I am happy"
    prediction, confidence = predictor.output_emotion(text)
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
