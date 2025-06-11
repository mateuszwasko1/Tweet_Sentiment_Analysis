import os
import pandas as pd
import emoji
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import re

class MainPreprocessing():
    def __init__(self):
        self._label_encoder = LabelEncoder()
        self.at_inference = False

    def _extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                 label_name: str) -> tuple[pd.DataFrame,
                                                           pd.DataFrame]:
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def _translate_emoji(self, text: str) -> str:
        return emoji.demojize(text)

    def clean_text(
            self,
            text: str,) -> str:
        text = self._translate_emoji(text)
        text = BeautifulSoup(text, "lxml").get_text()
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _preprocess_training_df(self, df: pd.DataFrame, training=False):
        X, y = self._extract_features_labels(df, "tweet", "emotion")
        X = X.apply(lambda text: self.clean_text(text,))
        if training:
            y = self._label_encoder.fit_transform(y)
        else:
            y = self._label_encoder.transform(y)
        return X, y

    def _preprocess_df(self, df: pd.DataFrame):
        X = df["tweet"]
        X = X.apply(lambda text: self.clean_text(text))
        return X

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
            X_training, y_training = self._preprocess_training_df(training_data, training=True)
            X_dev, y_dev = self._preprocess_training_df(dev_data)
            X_test, y_test = self._preprocess_training_df(test_data)
            return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)
        else:
            if data is None:
                raise ValueError("Data must be provided for inference.")
            data = pd.DataFrame({"tweet": [data]})
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame.")
            preprocessed_df = self._preprocess_df(data)
            return preprocessed_df


if __name__ == "__main__":
    preprocessor = MainPreprocessing()
    ((X_training, y_training),
     (X_dev, y_dev), (X_test, y_test)) = preprocessor.preprocessing_pipeline()

    print(X_training[7])
