import pandas as pd
import os


class DataLoading():
    def __init__(self):
        """
        Stores the relative paths of the dowloaded data and desired path for
        saving the converted and merged data.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_input = os.path.join(script_dir, "..", "..", "data",
                                       "original", "EI-reg")
        self.base_output = os.path.join(script_dir, "..", "..", "data", "raw")
        self.emotions = ["anger", "fear", "sadness", "joy"]

    def merge_emotions_to_df(self, dataset: str) -> tuple[pd.DataFrame,
                                                          os.PathLike]:
        """
        Loads all emotion txt files into one DataFrame.

        Args:
            dataset (str): The name of the desired dataset which emotions will
            be merged.

        Returns:
            pd.DataFrame: The merged emotions of the given dataset
            os.PathLike: The path where a json file of the created DataFrame
            can be stored

        Raises:
            ValueError: If the passed dataset is not in the allowed set of
            strings.
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

    def loading_pipeline(self):
        """
        Loads all the merged datasets into JSON files for later retrieval.
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
