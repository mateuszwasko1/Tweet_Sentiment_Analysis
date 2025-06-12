from project_name.preprocessing.baseline_preprocessing import BaselinePreprocessor
import numpy as np
import joblib
import pickle
from tqdm import tqdm


class PredictBaseline():
    def __init__(self):
        baseline_model_path = "/kaggle/input/bias012345678901234/biasprediction/models/baseline_model"
        vectoriser_path = "/kaggle/input/bias012345678901234/biasprediction/models/tfidf_vectorizer"
        with open(baseline_model_path, "rb") as f:
            self.model = pickle.load(f)
        self.vectoriser = joblib.load(vectoriser_path)
        self.preprocessor = BaselinePreprocessor()

    def predict(self, texts):
        print("cleaning texts ...")
        preprocessed_texts = [self.preprocessor.clean_text(text) for text in tqdm(texts)]

        print("vectorising...")
        vector = self.vectoriser.transform(preprocessed_texts)

        print("predicting...")
        probabilities = self.model.predict_proba(vector)
        predictions = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)

        return predictions, confidence_scores