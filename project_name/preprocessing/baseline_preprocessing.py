import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy import sparse


class BaselinePreprocessor():
    def __init__(self, use_tfidf: bool = True):
        self.use_tfidf = use_tfidf

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str, second_label_name=False
                                ) -> tuple[pd.DataFrame, pd.DataFrame,
                                           pd.DataFrame]:
        # Add value error to check if allowed names are passed on
        X = df[feature_name]
        y = df[label_name]
        z = df[second_label_name] if second_label_name else None
        return X, y, z

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'@\w+', 'user', text)
        return text

    def vectorize(self, X: pd.DataFrame, use_tfidf: bool = True):
        if use_tfidf is True:
            if self.test_data is True:
                X_vec = self.vectorizer.transform(X)
            else:
                self.vectorizer = TfidfVectorizer()
                X_vec = self.vectorizer.fit_transform(X)
        else:
            glove_df = pd.read_csv("data/glove.twitter.27B.100d.txt", sep=" ", index_col=0, header=None, quoting=3)
            glove_model = {word: row.values for word, row in glove_df.iterrows()}

            X_vec = np.vstack([self.glove(tweet, glove_model) for tweet in X])

        return X_vec

    def glove(self, tweet, glove_model):
        words = tweet.split()
        vectors = [glove_model[word] for word in words if word in glove_model]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100)
        
    def one_hot_encoder(self, labels: pd.DataFrame):
        ohe = OneHotEncoder()
        embedings = ohe.fit_transform(labels.to_frame())
        return embedings

    def preprocess_df(self, df: pd.DataFrame):
        if self.use_tfidf:
            X, y, _= self.extract_features_labels(df, "tweet", "emotion")
        else:
            X, y, z = self.extract_features_labels(df, "tweet", "score", "emotion")
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, use_tfidf=self.use_tfidf)
        X_vec = hstack([X_vec, self.one_hot_encoder(z)]) if not self.use_tfidf else X_vec
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
        self.test_data = False
        X_training, y_training = self.preprocess_df(pd.concat([training_data, dev_data]))
        # added +dev_data in line above and deleted (X_dev, y_dev) below
        #X_dev, y_dev = self.preprocess_df(dev_data)
        self.test_data = True
        X_test, y_test = self.preprocess_df(test_data)
        return (X_training, y_training), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = BaselinePreprocessor()
    (X_training,
     y_training), (X_test, y_test) = preprocessor.preprocessing_pipeline()
    print(X_training, y_training)
    #deleted (X_dev, y_dev) above
    #print(X_dev, y_dev)
    print(X_test, y_test)
