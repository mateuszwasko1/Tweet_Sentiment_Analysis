import os
import pandas as pd
import string
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import emoji
from cleantext import clean
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re


class MainPreprocessing():
    def __init__(self, test_data: bool = False):
        self.test_data = test_data
        self.label_encoder = LabelEncoder()
        self.processor = TextPreProcessor(normalize=[
            'url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'], annotate={"hashtag", "allcaps",
                                                 "elongated", "repeated",
                                                 'emphasis', 'censored'},
            segmenter="twitter",
            corrector="twitter", unpack_contractions=True,
            spell_correct_elong=True,
            unpack_hashtags=True, dicts=[emoticons])
        # tokenizer=SocialTokenizer(lowercase=True).tokenize)

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
        # Add value error to check if allowed names are passed on
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def apply_clean_text(self, text: str) -> str:
        return clean(
            text, to_ascii=True,
            normalize_whitespace=True,
            no_line_breaks=False,
            strip_lines=True,
            keep_two_line_breaks=False,
            )

    def use_ekphrasis(self, text: str) -> str:
        return self.processor.pre_process_doc(text)

    def translate_emoji(self, text: str) -> str:
        return emoji.demojize(text)

    def remove_punctuation(self, tokens: list[str]) -> str:
        punctuation = set(string.punctuation) - {"!"} - {"?"}
        without_punctuation = [token for token in tokens if
                               token not in punctuation]
        return without_punctuation

    def remove_stopwords(self, text:str) -> str:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered_words)

    def clean_text(
            self,
            text: str,
            ekphrasis_preprocessing: bool) -> str:
        text = text.lower()
        text = BeautifulSoup(text, "lxml").get_text()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        #text = text.replace(":", " ")
        #text = text.replace("\\n", " ")
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = self.remove_stopwords(text)
        if ekphrasis_preprocessing:
            # text = self.translate_emoji(text)
            text = self.use_ekphrasis(text)
            # tokens = self.remove_punctuation(tokens)
            text = self.apply_clean_text(text)
        return text

    def preprocess_df(
            self,
            df: pd.DataFrame,
            ekphrasis_preprocessing: bool,
            training=False):
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        X = X.apply(lambda text: self.clean_text(text,
                                                 ekphrasis_preprocessing))
        if training:
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)
        return X, y

    def preprocessing_pipeline(self, ekphrasis_preprocessing=True):
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
        X_training, y_training = self.preprocess_df(training_data,
                                                    training=True)
        X_dev, y_dev = self.preprocess_df(dev_data)
        X_test, y_test = self.preprocess_df(test_data)
        return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = MainPreprocessing()
    ((X_training, y_training),
     (X_dev, y_dev), (X_test, y_test)) = preprocessor.preprocessing_pipeline()

    print(X_training[7])
