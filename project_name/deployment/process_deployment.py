import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
from project_name.preprocessing.bert_preprocessing import (
    MainPreprocessing)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import torch.nn.functional as F
from fastapi import FastAPI
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

app = FastAPI()


class PredictEmotion():
    def __init__(self, baseline=False):
        self.baseline = baseline
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if baseline:
            model_loader = ModelSaver()
            self.model = model_loader.load_model("baseline_model").to(self.device)
            self.preprocessor = BaselinePreprocessor()
        else:  # Use BERT model
            bert_model_path = "models/saved_bert/model"
            bert_label_encoder_path = "models/saved_bert/" \
                                      "label_encoder"
            self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_model_path).to(self.device)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                bert_model_path)
            self.label_encoder = joblib.load(bert_label_encoder_path)
            self.preprocessor = MainPreprocessing()

    def predict(self, text, batch_size=32):
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
            
            dataset = TensorDataset(train_encodings["input_ids"],
                                    train_encodings["attention_mask"])
            dataloader = DataLoader(dataset, batch_size=batch_size)

            all_predictions = []
            all_confs = []

            with torch.no_grad():
                for input_ids, attention_mask in tqdm(dataloader, desc="batch prediction"):
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)

                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask).logits

                    probability = F.softmax(logits, dim=1)

                    prob_val, predicted_class = torch.max(probability, dim=1)
                    all_confs.extend(prob_val.cpu().tolist())
                    all_predictions.extend(
                        self.label_encoder.inverse_transform(predicted_class.cpu())
                        )
            return all_predictions[0], all_confs[0]

    def output_emotion(self, text: str) -> str:
        tweet_cleaned = self.preprocessor.preprocessing_pipeline(
            at_inference=True, data=text)
        if hasattr(tweet_cleaned, "iloc"):
            cleaned_text = tweet_cleaned.iloc[0]
        else:
            cleaned_text = tweet_cleaned
        prediction, confidence = self.predict(cleaned_text)
        if isinstance(prediction, (np.ndarray, list)):
            return str(prediction[0]), float(round(confidence, 2))
        return str(prediction), float(round(confidence, 2))


if __name__ == "__main__":
    predictor = PredictEmotion(baseline=False)
    text = "I am happy"
    prediction, confidence = predictor.output_emotion(text)
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
