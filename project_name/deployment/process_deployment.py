import os
import sys

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
from project_name.preprocessing.bert_preprocessing import MainPreprocessing

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

app = FastAPI()


class PredictEmotion:
    """
    Interface to classify emotions in text using either a baseline
    scikit-learn model or a BERT-based transformer model.
    """

    def __init__(self, use_baseline: bool = False) -> None:
        """
        Load the chosen model and its preprocessing pipeline.

        Args:
            use_baseline (bool): If True, loads a scikit-learn
                baseline model. Otherwise loads a HuggingFace
                BERT sequence classification model.

        Raises:
            FileNotFoundError: If expected model files are missing.
        """
        self.use_baseline = use_baseline

        if use_baseline:
            saver = ModelSaver()
            self.model: ClassifierMixin = saver.load_model("baseline_model")
            self.preprocessor = BaselinePreprocessor()
        else:
            bert_dir = "models/saved_bert/model"
            label_enc_file = "models/saved_bert/label_encoder"

            self.model = AutoModelForSequenceClassification.from_pretrained(
                bert_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.label_encoder: LabelEncoder = joblib.load(label_enc_file)
            self.preprocessor = MainPreprocessing()

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict emotion label and confidence score for preprocessed text.

        Args:
            text (str): Token or string input ready for the model.

        Returns:
            tuple[str, float]: (predicted_label, confidence_score)
        """
        if self.use_baseline:
            # baseline model expects array-like input
            preds = self.model.predict([text])
            probs = self.model.predict_proba([text])
            confidence = float(np.max(probs, axis=1)[0])
            label = str(preds[0])
        else:
            # tokenize and run through BERT
            batch = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                confidence_tensor, idx_tensor = torch.max(probs, dim=1)
                confidence = confidence_tensor.item()
                label = self.label_encoder.inverse_transform(
                    [idx_tensor.item()]
                )[0]

        return label, confidence

    def output_emotion(self, raw_text: str) -> tuple[str, float]:
        """
        Full pipeline: preprocess raw text, predict label, round confidence.

        Args:
            raw_text (str): Original text input (e.g., a tweet).

        Returns:
            tuple[str, float]: (predicted_label,
            confidence_rounded_to_two_decimals)
        """
        processed = self.preprocessor.preprocessing_pipeline(
            at_inference=True,
            data=raw_text,
        )

        # Handle pandas Series vs plain string
        if hasattr(processed, "iloc"):
            cleaned = str(processed.iloc[0])
        else:
            cleaned = str(processed)

        label, score = self.predict(cleaned)
        return label, round(score, 2)


if __name__ == "__main__":
    predictor = PredictEmotion(use_baseline=False)
    test_sentence = "I am happy"
    emotion, conf = predictor.output_emotion(test_sentence)
    print(f"Prediction: {emotion}, Confidence: {conf:.2f}")
