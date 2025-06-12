import pandas as pd
import os
from typing import Tuple, List


class DataLoading():
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
        Stores the relative paths of the dowloaded data and desired path for
        saving the converted and merged data.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_input = os.path.join(script_dir, "..", "..", "data",
                                       "original", "EI-reg")
        self.base_output: str = os.path.join(
            script_dir, "..", "..", "data", "raw"
        )
        self.emotions: List[str] = ["anger", "fear", "sadness", "joy"]

    def merge_emotions_to_df(self, dataset: str) -> Tuple[pd.DataFrame,
                                                          os.PathLike]:
        """
        Loads all emotion txt files into one DataFrame.

        Args:
            dataset (str): The name of the desired dataset which emotions will
            be merged.

        Returns:
            Tuple[pd.DataFrame, os.PathLike]:
                - merged_df: DataFrame of all emotions
                - json_path: path where the JSON should be saved

        Raises:
            FileNotFoundError: If any of the expected emotion
            files are missing.
        """
        valid_datasets = ["development", "test-gold", "training"]
        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset '{dataset}'. Expected one of\
                              {valid_datasets}.")

        all_data = []
        if dataset == "development":
            abbreviation = "dev"
        elif dataset == "training":
            abbreviation = "train"
        else:
            abbreviation = "test-gold"

        for emotion in self.emotions:
            filename = f"2018-EI-reg-En-{emotion}-{abbreviation}.txt"
            filepath = os.path.join(self.base_input, dataset, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filepath}")

            df = pd.read_csv(filepath, sep="\t", usecols=["Tweet",
                                                          "Affect Dimension",
                                                          "Intensity Score"])
            df.columns = ["tweet", "emotion", "score"]
            all_data.append(df)

        merged_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(self.base_output, exist_ok=True)
        json_path = os.path.join(self.base_output, f"{dataset}_merged.json")

        return merged_df, json_path

    def loading_pipeline(self) -> None:
        """
        Runs the merge on all splits and writes each merged DataFrame to JSON.

        Writes files:
            data/raw/development_merged.json
            data/raw/test-gold_merged.json
            data/raw/training_merged.json

        Returns:
            None
        """
        datasets = ["development", "test-gold", "training"]
        loading = DataLoading()
        for i in datasets:
            merged_df, json_path = loading.merge_emotions_to_df(i)
            merged_df.to_json(json_path, orient="records", lines=True,
                              force_ascii=False)


if __name__ == "__main__":
    loading = DataLoading()
    loading.loading_pipeline()
