import os
import re
import pandas as pd
import emoji
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder


class MainPreprocessing:
    """
    Preprocess text data for BERT-based emotion classification. Handles
    emoji translation, HTML cleaning, token cleaning, label encoding,
    and training vs. inference pipelines.
    """

    def __init__(self) -> None:
        """
        Initialize the preprocessing pipeline with a LabelEncoder.
        """
        self._label_encoder: LabelEncoder = LabelEncoder()
        self.at_inference: bool = False

    def _extract_features_labels(
        self,
        df: pd.DataFrame,
        feature_name: str,
        label_name: str,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Extract feature and label series from a DataFrame.

        Args:
            df: DataFrame with feature and label columns.
            feature_name: Column key for features (text).
            label_name: Column key for labels (classes).

        Returns:
            A tuple (X, y) where X is the text series and
            y is the label series.
        """
        X: pd.Series = df[feature_name]
        y: pd.Series = df[label_name]
        return X, y

    def _translate_emoji(self, text: str) -> str:
        """
        Convert unicode emojis in text to their text representations.

        Args:
            text: Input string potentially containing emojis.

        Returns:
            String with emojis replaced by text codes (e.g., ':smile:').
        """
        return emoji.demojize(text)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize a text string: translate emojis, remove HTML,
        strip user mentions, hashtags, non-alphanumeric chars,
        and extra spaces.

        Args:
            text: Raw text input.

        Returns:
            Cleaned text string.
        """
        text = self._translate_emoji(text)
        text = BeautifulSoup(text, "lxml").get_text()
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _preprocess_training_df(
        self,
        df: pd.DataFrame,
        training: bool = False,
    ) -> tuple[pd.Series, pd.Series | pd.Series]:
        """
        Clean text and encode labels for training or evaluation.

        Args:
            df: DataFrame containing 'tweet' and 'emotion' columns.
            training: If True, fit the LabelEncoder; otherwise transform only.

        Returns:
            Tuple of (cleaned text series, encoded labels array).
        """
        X, y = self._extract_features_labels(df, "tweet", "emotion")
        X_clean: pd.Series = X.apply(lambda t: self.clean_text(t))
        if training:
            y_encoded = pd.Series(
                self._label_encoder.fit_transform(y),
                index=y.index,
            )
        else:
            y_encoded = pd.Series(
                self._label_encoder.transform(y),
                index=y.index,
            )
        return X_clean, y_encoded

    def _preprocess_df(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Clean text for inference without label encoding.

        Args:
            df: DataFrame with a 'tweet' column.

        Returns:
            Series of cleaned tweet text.
        """
        return df["tweet"].apply(lambda t: self.clean_text(t))

    def preprocessing_pipeline(
        self,
        at_inference: bool = False,
        data: str | None = None,
    ):
        """
        Run the full preprocessing workflow for training or inference.

        Args:
            at_inference: If False, loads raw JSON files and returns
                          splits for train/dev/test; if True, expects
                          a single text string in 'data'.
            data: Raw text string for inference (required if at_inference).

        Returns:
            For training: ((X_train, y_train), (X_dev, y_dev),
            (X_test, y_test)).
            For inference: Series of cleaned text for the input string.

        Raises:
            ValueError: If 'data' is None when at_inference is True.
            TypeError: If 'data' is not a string when required.
        """
        self.at_inference = at_inference

        if not at_inference:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_path = os.path.join(
                script_dir,
                "..",
                "..",
                "data",
                "raw",
                "training_merged.json",
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
                script_dir,
                "..",
                "..",
                "data",
                "raw",
                "test-gold_merged.json",
            )

            train_df = pd.read_json(train_path, orient="records", lines=True)
            dev_df = pd.read_json(dev_path, orient="records", lines=True)
            test_df = pd.read_json(test_path, orient="records", lines=True)

            X_train, y_train = self._preprocess_training_df(
                pd.concat([train_df, dev_df]),
                training=True,
            )
            X_dev, y_dev = self._preprocess_training_df(dev_df)
            X_test, y_test = self._preprocess_training_df(test_df)
            return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

        if data is None:
            raise ValueError("Data must be provided for inference.")
        if not isinstance(data, str):
            raise TypeError("Data must be a string.")

        df = pd.DataFrame({"tweet": [data]})
        return self._preprocess_df(df)


if __name__ == "__main__":
    preprocessor = MainPreprocessing()
    (
        (X_train, y_train),
        (X_dev,   y_dev),
        (X_test,  y_test),
    ) = preprocessor.preprocessing_pipeline()
    print(X_train.iloc[7])
