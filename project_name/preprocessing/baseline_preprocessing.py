import os
import re
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from project_name.models.save_load_model import ModelSaver


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
        self.vectorizer: TfidfVectorizer | None = None
        self.at_inference: bool = False
        self.test_data: bool = False
        self.model_saver: ModelSaver = ModelSaver()

    def extract_features_labels(
        self,
        df: pd.DataFrame,
        feature_name: str,
        label_name: str,
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
        X: pd.Series = df[feature_name]
        y: pd.Series = df[label_name]
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
        cleaned = text.lower()
        cleaned = re.sub(r"@\w+", "user", cleaned)
        return cleaned

    def vectorize(
        self,
        X: pd.Series,
        fit: bool = True,
    ) -> scipy.sparse.spmatrix:
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
            vec = TfidfVectorizer()
            X_vec = vec.fit_transform(X)
            self.model_saver.save_model(vec, "tfidf_vectorizer")
            self.vectorizer = self.model_saver.load_model("tfidf_vectorizer")
        else:
            if self.vectorizer is None:
                self.vectorizer = self.model_saver.load_model(
                    "tfidf_vectorizer"
                )
            X_vec = self.vectorizer.transform(X)
        return X_vec

    def preprocess_training_df(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> tuple[scipy.sparse.spmatrix, pd.Series]:
        """
        Clean and vectorize training (or dev/test) DataFrame.

        Args:
            df: DataFrame with 'tweet' and 'emotion' columns.
            fit: Whether to fit vectorizer on this data.

        Returns:
            Tuple of TF-IDF matrix and label series.
        """
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        X_clean = X.apply(self.clean_text)
        X_vec = self.vectorize(X_clean, fit=fit)
        return X_vec, y

    def preprocess_df(
        self,
        df: pd.DataFrame,
    ) -> scipy.sparse.spmatrix:
        """
        Clean and vectorize a DataFrame for inference.

        Args:
            df: DataFrame with 'tweet' column.

        Returns:
            TF-IDF feature matrix for inference.
        """
        self.test_data = True
        X = df["tweet"].apply(self.clean_text)
        return self.vectorize(X, fit=False)

    def preprocessing_pipeline(
        self,
        at_inference: bool = False,
        data: str | None = None,
    ):
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

        if not at_inference:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_path = os.path.join(
                script_dir, "..", "..", "data", "raw", "training_merged.json"
            )
            dev_path = os.path.join(
                script_dir,
                "..",
                "..",
                "data",
                "raw",
                "development_merged.json",
            )
            test_path = os.path.join(
                script_dir, "..", "..", "data", "raw", "test-gold_merged.json"
            )

            train_df = pd.read_json(train_path, orient="records", lines=True)
            dev_df = pd.read_json(dev_path, orient="records", lines=True)
            test_df = pd.read_json(test_path, orient="records", lines=True)

            X_train, y_train = self.preprocess_training_df(
                pd.concat([train_df, dev_df]), fit=True
            )
            X_dev, y_dev = self.preprocess_training_df(dev_df, fit=False)
            X_test, y_test = self.preprocess_training_df(test_df, fit=False)
            self.test_data = True

            return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)
        else:
            if data is None:
                raise ValueError("Data must be provided for inference.")
            if not isinstance(data, str):
                raise TypeError("Data must be a string.")
            df = pd.DataFrame({"tweet": [data]})
            return self.preprocess_df(df)


if __name__ == "__main__":
    preprocessor = BaselinePreprocessor()
    (X_train, y_train), (X_dev, y_dev), (
        X_test, y_test) = preprocessor.preprocessing_pipeline()
    print(X_train.shape, y_train.shape)
    print(X_dev.shape, y_dev.shape)
    print(X_test.shape, y_test.shape)
