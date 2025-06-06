import os
import pandas as pd
import emoji
from cleantext import clean
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class MainPreprocessing():
    def __init__(self, test_data: bool = False):
        self._label_encoder = LabelEncoder()

    def _extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def _apply_clean_text(self, text: str) -> str:
        return clean(
            text, to_ascii=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=True,
            keep_two_line_breaks=False,
            )

    def _translate_emoji(self, text: str) -> str:
        return emoji.demojize(text)

    def _remove_stopwords(self, text: str) -> str:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [
            word for word in words if word.lower() not in stop_words]
        return " ".join(filtered_words)

    def clean_text(
            self,
            text: str,) -> str:
        text = self._translate_emoji(text)
        text = text.lower()
        text = BeautifulSoup(text, "lxml").get_text()
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = self._remove_stopwords(text)
        text = self._apply_clean_text(text)
        return text

    def _preprocess_df(
            self,
            df: pd.DataFrame,
            training=False):
        X, y = self._extract_features_labels(df, "tweet", "emotion")
        X = X.apply(lambda text: self.clean_text(text,))
        if training:
            y = self._label_encoder.fit_transform(y)
        else:
            y = self._label_encoder.transform(y)
        return X, y

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
        X_training, y_training = self._preprocess_df(
            training_data, training=True)
        X_dev, y_dev = self._preprocess_df(dev_data)
        X_test, y_test = self._preprocess_df(test_data)
        return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = MainPreprocessing()
    ((X_training, y_training),
     (X_dev, y_dev), (X_test, y_test)) = preprocessor.preprocessing_pipeline()

    print(X_training[7])
