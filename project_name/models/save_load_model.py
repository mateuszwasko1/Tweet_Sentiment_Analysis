import os
import pickle


class ModelSaver:
    """
    Provides utilities to save and load Python objects (models) via pickle.

    Attributes:
        path (str): Filesystem directory where models are stored.
    """

    def __init__(self) -> None:
        """
        Initialize ModelSaver by setting the model storage directory
        relative to this script's location.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path: str = os.path.join(
            script_dir, os.pardir, os.pardir, "models"
        )

    def save_model(self, model: object, file_name: str) -> None:
        """
        Serialize and save a model object to disk using pickle.

        Args:
            model: The Python object to serialize (e.g., a trained model).
            file_name: Name of the file under the storage directory.

        Raises:
            OSError: If the target directory does not exist or is not writable.
        """
        file_path = os.path.join(self.path, file_name)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_path}")

    def load_model(self, file_name: str) -> object:
        """
        Load and deserialize a model object from disk using pickle.

        Args:
            file_name: Name of the file under the storage directory.

        Returns:
            The deserialized Python object (e.g., a trained model).

        Raises:
            FileNotFoundError: If the specified file is not present.
            OSError: If the file cannot be read.
        """
        file_path = os.path.join(self.path, file_name)
        if file_name not in os.listdir(self.path):
            raise FileNotFoundError(
                f"Model file '{file_name}' not found in "
                f"{self.path}."
            )
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return model
