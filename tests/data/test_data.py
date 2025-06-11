import os
import json
import unittest
from typing import List
import pandas as pd

from project_name.data.data_loading import DataLoading


class DataLoadingTest(unittest.TestCase):
    """
    Unit tests for the DataLoading class, ensuring JSON merging
    and file access behave as expected.
    """

    def setUp(self) -> None:
        """
        Initialize a DataLoading instance before each test.
        """
        self.loader: DataLoading = DataLoading()

    def test_merge_emotions_to_df(self) -> None:
        """
        Test that merge_emotions_to_df returns a non-empty DataFrame
        with required columns for a valid dataset name.
        """
        dataset: str = "development"
        df: pd.DataFrame
        path: str

        df, path = self.loader.merge_emotions_to_df(dataset)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn("tweet", df.columns)
        self.assertIn("emotion", df.columns)
        self.assertIn("score", df.columns)

    def test_invalid_dataset_name(self) -> None:
        """
        Test that an invalid dataset name raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            self.loader.merge_emotions_to_df("invalid-dataset")
        self.assertIn("Invalid dataset", str(context.exception))

    def test_loading_pipeline(self) -> None:
        """
        Test that merge_emotions_to_df writes and returns a valid JSON path,
        and that the JSON file contains at least one record.
        """
        for dataset in ["development", "training", "test-gold"]:
            df, json_path = self.loader.merge_emotions_to_df(dataset)
            self.assertTrue(
                os.path.exists(json_path),
                f"JSON path should exist: {json_path}",
            )

            data: List[dict] = []
            with open(json_path, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line))

            self.assertGreater(
                len(data),
                0,
                "JSON file should contain at least one record",
            )
            self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
