import unittest
import pandas as pd
import pandas.testing as pd_t

from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)


class PreprocessingTest(unittest.TestCase):
    """
    Unit tests for BaselinePreprocessor to verify text feature extraction,
    text cleaning, vectorization behavior, and DataFrame preprocessing.
    """

    def setUp(self) -> None:
        """
        Initialize a BaselinePreprocessor instance for testing.
        """
        self.processor: BaselinePreprocessor = BaselinePreprocessor()

    def test_extract_feature_labels(self) -> None:
        """
        Verify that extract_features_labels returns correct Series X and y
        from a DataFrame with 'tweet' and 'emotion' columns.
        """
        data: dict[str, list[str]] = {
            "tweet": ["This is an angry tweet", "This is a sad tweet"],
            "emotion": ["angry", "sadness"],
        }
        df: pd.DataFrame = pd.DataFrame(data)

        X, y = self.processor.extract_features_labels(df, "tweet", "emotion")

        expected_X: pd.Series = pd.Series(
            ["This is an angry tweet", "This is a sad tweet"],
            name="tweet",
        )
        pd_t.assert_series_equal(X, expected_X)

        expected_y: pd.Series = pd.Series(["angry", "sadness"], name="emotion")
        pd_t.assert_series_equal(y, expected_y)

    def test_clean_text(self) -> None:
        """
        Confirm that clean_text lowercases text
        and replaces mentions with 'user'.
        """
        raw_text: str = "HELLO @world, how are you"
        cleaned: str = self.processor.clean_text(raw_text)
        expected: str = "hello user, how are you"
        self.assertEqual(cleaned, expected)

    def test_vectorize(self) -> None:
        """
        Ensure vectorize produces a TF-IDF matrix of expected shape
        for training data, and handles test data transform correctly.
        """
        X_train: pd.Series = pd.Series(
            ["This is an angry tweet", "This is a sad tweet"],
            name="tweet",
        )
        X_train_vec = self.processor.vectorize(X_train)
        self.assertEqual(X_train_vec.shape[0], 2)

        self.processor.at_inference = True
        X_test: pd.Series = pd.Series([
            "This is an angry tweet in test data set"
        ], name="tweet")
        X_test_vec = self.processor.vectorize(X_test, fit=False)
        self.assertEqual(X_test_vec.shape[0], 1)

    def test_preprocess_df(self) -> None:
        """
        Validate that preprocess_df returns a TF-IDF matrix for inference data.
        """
        data: dict[str, list[str]] = {
            "tweet": ["Neutral tweet text"],
            "emotion": ["joy"],
        }
        df: pd.DataFrame = pd.DataFrame(data)

        result = self.processor.preprocess_df(df)
        self.assertEqual(result.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
