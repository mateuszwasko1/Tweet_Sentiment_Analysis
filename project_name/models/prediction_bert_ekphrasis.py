from project_name.preprocessing.ekphrasis_preprocessing import (
    MainPreprocessing)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import torch.nn.functional as F
from fastapi import FastAPI


class PredictEkphrasisBert():
    def __init__(self):
        bert_model_path = "models/saved_bert/model"
        bert_label_encoder_path = "models/saved_bert/" \
                                  "label_encoder"
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            bert_model_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.label_encoder = joblib.load(bert_label_encoder_path)

    def predict(self, text):
        preprocessing = MainPreprocessing()
        preprocessed_text = preprocessing.clean_text(text, False)

        train_encodings = self.bert_tokenizer(
            preprocessed_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt")

        with torch.no_grad():
            logits = self.bert_model(**train_encodings).logits
            probability = F.softmax(logits, dim=1)

            prob_val, predicted_class = torch.max(probability, dim=1)
            predicted_label = self.label_encoder.inverse_transform(
                [predicted_class.item()])[0]
            confidence = prob_val.item()

        # return preprocessed_text #(predicted_label, confidence)

        return (predicted_label, confidence)
