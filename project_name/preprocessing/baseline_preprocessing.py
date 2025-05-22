import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BaselinePreprocessor():
    def __init__(self, test_data: bool = False):
        self.test_data = test_data

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
        # Add value error to check if allowed names are passed on
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'@\w+', 'user', text)
        return text

    def vectorize(self, X: pd.DataFrame, use_tfidf: bool = True):
        vectorizer = TfidfVectorizer()
        if use_tfidf is True:
            if self.test_data is True:
                X_vec = vectorizer.transform(X)
            else:
                X_vec = vectorizer.fit_transform(X)
        # else: Glove Implementation
        return X_vec

    def preprocess_df(self, df: pd.DataFrame):
        X, y = self.extract_features_labels(df, "Tweet", "Emotion")
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, use_tfidf=True)
        return X_vec, y

    def preprocessing_pipeline(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                  "training_merged.json")
        dev_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                "development_merged.json")
        test_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                 "test-gold_merged.json")
        training_data = pd.read_json(train_path, orient="records", lines=True)
        dev_data = pd.read_json(dev_path,
                                orient="records", lines=True)
        test_data = pd.read_json(test_path,
                                 orient="records", lines=True)
        X_training, y_training = self.preprocess_df(training_data)
        X_dev, y_dev = self.preprocess_df(dev_data)
        X_test, y_test = self.preprocess_df(test_data)
        return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = BaselinePreprocessor()
    (X_training,
     y_training), (X_dev,
                   y_dev), (X_test,
                            y_test) = preprocessor.preprocessing_pipeline()
    print(X_training, y_training)
    print(X_dev, y_dev)
    print(X_test, y_test)
