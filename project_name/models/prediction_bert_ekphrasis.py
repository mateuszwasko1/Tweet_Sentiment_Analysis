from project_name.preprocessing.ekphrasis_preprocessing import (
    MainPreprocessing)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI


class PredictEkphrasisBert():
    def __init__(self):
        bert_model_path = "models/saved_bert/model"
        bert_label_encoder_path = "models/saved_bert/" \
                                  "label_encoder"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            bert_model_path).to(self.device)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.label_encoder = joblib.load(bert_label_encoder_path)
        self.preprocessing = MainPreprocessing()

    def predict(self, texts, batch_size=32):
        print("cleaning texts...")
        preprocessed_text = [self.preprocessing.clean_text(text)
                             for text in tqdm(texts)]

        train_encodings = self.bert_tokenizer(
            preprocessed_text,
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

                logits = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                probability = F.softmax(logits, dim=1)

                prob_val, predicted_class = torch.max(probability, dim=1)
                all_predictions.extend(
                    self.label_encoder.inverse_transform(predicted_class.cpu())
                    )
                all_confs.extend(prob_val.cpu().numpy())

        return (all_predictions, all_confs)
