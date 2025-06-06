import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from project_name.models.save_load_model import ModelSaver


class BaselinePreprocessor():
    def __init__(self):
        self.vectorizer = None
        # Stores whether the preprocessor is used for inference or training
        self.at_inference = False
        self.test_data = False
        self.model_saver = ModelSaver()

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str
                                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        text = text.lower()
        text = re.sub(r'@\w+', 'user', text)
        return text

    def vectorize(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            vectorizer = TfidfVectorizer()
            X_vec = vectorizer.fit_transform(X)
            self.model_saver.save_model(vectorizer, "tfidf_vectorizer")
            self.vectorizer = self.model_saver.load_model("tfidf_vectorizer")
        else:
            if self.vectorizer is None:
                self.vectorizer = self.model_saver.load_model(
                    "tfidf_vectorizer")
            X_vec = self.vectorizer.transform(X)
        return X_vec

    def preprocess_training_df(self, df: pd.DataFrame, fit=True):
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, fit=fit)
        return X_vec, y

    def preprocess_df(self, df: pd.DataFrame):
        self.test_data = True
        X = df["tweet"]
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, fit=False)
        return X_vec

    def preprocessing_pipeline(self, at_inference: bool = False, data=None):
        self.at_inference = at_inference
        if at_inference is False:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                      "training_merged.json")
            dev_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                    "development_merged.json")
            test_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                     "test-gold_merged.json")
            training_data = pd.read_json(train_path, orient="records",
                                         lines=True)
            dev_data = pd.read_json(dev_path, orient="records", lines=True)
            test_data = pd.read_json(test_path, orient="records", lines=True)
            X_training, y_training = self.preprocess_training_df(pd.concat(
                [training_data, dev_data]), fit=True)
            self.test_data = True
            X_dev, y_dev = self.preprocess_training_df(dev_data, fit=False)
            X_test, y_test = self.preprocess_training_df(test_data, fit=False)
            return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)
        else:
            if data is None:
                raise ValueError("Data must be provided for inference.")
            data = pd.DataFrame({"tweet": [data]})
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame.")
            preprocessed_df = self.preprocess_df(data)
            return preprocessed_df


if __name__ == "__main__":
    preprocessor = BaselinePreprocessor()
    (X_training, y_training), (X_dev, y_dev), (
        X_test, y_test) = preprocessor.preprocessing_pipeline()
    (X_training, y_training), (
        X_test, y_test) = preprocessor.preprocessing_pipeline()
    print(X_training, y_training)
    print(X_dev, y_dev)
    print(X_test, y_test)
