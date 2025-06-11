import pandas as pd


class BERTweetPreprocessor:
    """
    Preprocess dataset for fine-tuning and evaluation of BERTweet models.
    Handles feature/label extraction, label encoding,
    and dataframe preprocessing.
    """

    def extract_features_labels(
        self,
        df: pd.DataFrame,
        feature_name: str,
        label_name: str,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Extract feature and label series from a DataFrame.

        Args:
            df: DataFrame containing data.
            feature_name: Column name for input texts.
            label_name: Column name for target labels.

        Returns:
            A tuple (X, y) where:
              - X: Series of input texts.
              - y: Series of label values.
        """
        X: pd.Series = df[feature_name]
        y: pd.Series = df[label_name]
        return X, y

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map string emotion labels to integer codes.

        Args:
            df: DataFrame with an 'emotion' column.

        Returns:
            DataFrame with 'emotion' column encoded as integers.

        Raises:
            ValueError: If any label is not recognized.
        """
        emotion_map: dict[str, int] = {
            "anger": 0,
            "fear": 1,
            "joy": 2,
            "sadness": 3,
        }
        df_copy: pd.DataFrame = df.copy()
        df_copy["emotion"] = df_copy["emotion"].map(emotion_map)
        if df_copy["emotion"].isnull().any():
            raise ValueError("DataFrame contains invalid emotion labels.")
        return df_copy

    def preprocess_df(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Apply label encoding and extract features and labels for modeling.

        Args:
            df: DataFrame containing 'tweet' and 'emotion' columns.

        Returns:
            A tuple (X, y) where:
              - X: Series of raw tweet texts.
              - y: Series of encoded emotion labels.
        """
        df_encoded: pd.DataFrame = self.encode_labels(df)
        X, y = self.extract_features_labels(
            df_encoded,
            "tweet",
            "emotion",
        )
        return X, y
