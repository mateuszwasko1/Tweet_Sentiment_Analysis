import pandas as pd
import os
from typing import List, Tuple


class DataLoading:
    """
    Class for loading and merging emotion datasets from raw text files.

    Attributes:
        base_input (str): Path to the folder containing original emotion files.
        base_output (str): Path to the folder where
        merged JSON files will be saved.
        emotions (List[str]): List of emotion labels to process.
    """

    def __init__(self) -> None:
        """
        Initialize the DataLoading instance with default input/output paths
        and a predefined list of emotion categories.
        """
        script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.base_input: str = os.path.join(
            script_dir, "..", "..", "data", "original", "EI-reg"
        )
        self.base_output: str = os.path.join(
            script_dir, "..", "..", "data", "raw"
        )
        self.emotions: List[str] = ["anger", "fear", "sadness", "joy"]

    def merge_emotions_to_df(self, dataset: str) -> Tuple[pd.DataFrame, str]:
        """
        Load all emotion-specific text files for a given dataset and merge them
        into a single pandas DataFrame.

        Args:
            dataset (str): Name of the dataset to load. Must be one of
                'development', 'test-gold', or 'training'.

        Returns:
            Tuple[pd.DataFrame, str]:
                - pd.DataFrame: DataFrame containing columns
                ['tweet', 'emotion', 'score']
                  for all emotions in the dataset.
                - str: File path where the merged DataFrame
                should be saved as JSON.

        Raises:
            ValueError: If `dataset` is not a supported name.
            FileNotFoundError: If any of the expected emotion
            files are missing.
        """
        valid_datasets: List[str] = ["development", "test-gold", "training"]
        if dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dtaset '{dataset}'.Expected one of {valid_datasets}."
            )

        abbreviation_map = {
            "development": "dev",
            "training": "train",
            "test-gold": "test-gold"
        }
        abbr: str = abbreviation_map[dataset]

        all_data: List[pd.DataFrame] = []
        for emotion in self.emotions:
            filename: str = f"2018-EI-reg-En-{emotion}-{abbr}.txt"
            filepath: str = os.path.join(self.base_input, dataset, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filepath}")

            df: pd.DataFrame = pd.read_csv(
                filepath,
                sep="\t",
                usecols=["Tweet", "Affect Dimension", "Intensity Score"],
            )
            df.columns = ["tweet", "emotion", "score"]
            all_data.append(df)

        merged_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)

        os.makedirs(self.base_output, exist_ok=True)
        json_path: str = os.path.join(
            self.base_output, f"{dataset}_merged.json"
        )

        return merged_df, json_path

    def loading_pipeline(self) -> None:
        """
        Run the full data loading and merging pipeline, saving each merged
        dataset as a JSON file in the output directory.
        """
        datasets: List[str] = ["development", "test-gold", "training"]
        for ds in datasets:
            df, json_path = self.merge_emotions_to_df(ds)
            df.to_json(
                json_path,
                orient="records",
                lines=True,
                force_ascii=False
            )


if __name__ == "__main__":
    loader = DataLoading()
    loader.loading_pipeline()
