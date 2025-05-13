import unittest
import os
import json
from project_name.data.data_loading import DataLoading


class DataLoadingTest(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoading()

    def test_merge_emotions_to_df(self):
        dataset = "development"
        with self.subTest(dataset=dataset):
            df, _ = self.loader.merge_emotions_to_df(dataset)
            self.assertFalse(df.empty)
            self.assertIn("tweet", df.columns)
            self.assertIn("emotion", df.columns)
            self.assertIn("score", df.columns)

    def test_invalid_dataset_name(self):
        with self.assertRaises(ValueError) as context:
            self.loader.merge_emotions_to_df("invalid-dataset")
        self.assertIn("Invalid dataset", str(context.exception))

    def test_loading_pipeline(self):
        for dataset in ["development", "training", "test-gold"]:
            df, json_path = self.loader.merge_emotions_to_df(dataset)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
                self.assertGreater(len(data), 0)
            self.assertIsInstance(data, list)
