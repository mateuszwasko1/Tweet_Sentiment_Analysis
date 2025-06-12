import re
import os
import scipy.sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tweet_sentiment_analysis.models.save_load_model import ModelSaver


class BaselinePreprocessor:
    """
    Preprocess text data for baseline emotion classification. Handles
    feature extraction, text cleaning, TF-IDF vectorization, and
    data pipeline for training and inference.
    """
    def __init__(self) -> None:
        """
        Initialize the preprocessor with default settings and a ModelSaver.
        """
        self.vectorizer = None
        # Stores whether the preprocessor is used for inference or training
        self.at_inference = False
        self.test_data = False
        self.model_saver = ModelSaver()

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str
                                ) -> tuple[pd.Series, pd.Series]:
        """
        Separate features and labels from a DataFrame.

        Args:
            df: DataFrame containing feature and label columns.
            feature_name: Column name for text features.
            label_name: Column name for target labels.

        Returns:
            A tuple (X, y) where X is the feature
            series and y is the label series.
        """
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def clean_text(self, text: str) -> str:
        """
        Normalize and anonymize a text string.

        Args:
            text: Raw text input.

        Returns:
            Cleaned, lowercase text with user mentions replaced.

        Raises:
            ValueError: If input is not a string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        text = text.lower()
        text = re.sub(r'@\w+', 'user', text)
        return text

    def vectorize(self,
                  X: pd.Series,
                  fit: bool = True) -> scipy.sparse.spmatrix:
        """
        Transform text data into TF-IDF feature vectors.

        Args:
            X: Series of cleaned text.
            fit: If True, fit a new vectorizer and save it;
                 otherwise load existing and transform.

        Returns:
            Sparse TF-IDF feature matrix.
        """
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

    def preprocess_training_df(self,
                               df: pd.DataFrame,
                               fit=True) -> tuple[scipy.sparse.spmatrix,
                                                  pd.Series]:
        """
        Clean and vectorize training (or dev/test) DataFrame.

        Args:
            df: DataFrame with 'tweet' and 'emotion' columns.
            fit: Whether to fit vectorizer on this data.

        Returns:
            Tuple of TF-IDF matrix and label series.
        """
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, fit=fit)
        return X_vec, y

    def preprocess_df(self, df: pd.DataFrame) -> scipy.sparse.spmatrix:
        """
        Clean and vectorize a DataFrame for inference.

        Args:
            df: DataFrame with 'tweet' column.

        Returns:
            TF-IDF feature matrix for inference.
        """
        self.test_data = True
        X = df["tweet"]
        X = X.apply(self.clean_text)
        X_vec = self.vectorize(X, fit=False)
        return X_vec

    def preprocessing_pipeline(self,
                               at_inference: bool = False,
                               data: str | None = None) -> (tuple[
                                   tuple[scipy.sparse.spmatrix, pd.Series],
                                   tuple[scipy.sparse.spmatrix, pd.Series],
                                   tuple[scipy.sparse.spmatrix, pd.Series]] |
                                   scipy.sparse.spmatrix):
        """
        Run end-to-end preprocessing for training or inference.

        Args:
            at_inference: If False, load raw JSON and return
                          train/dev/test splits; otherwise expects
                          a raw text string in 'data'.
            data: Raw text string for inference.

        Returns:
            If training: ((X_train, y_train), (X_dev, y_dev),
            (X_test, y_test)).
            If inference: TF-IDF matrix for the provided text.

        Raises:
            ValueError: If 'data' is missing when at_inference is True.
            TypeError: If 'data' is not a string.
        """
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
