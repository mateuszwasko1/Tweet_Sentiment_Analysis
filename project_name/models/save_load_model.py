import pickle
import os


class ModelSaver():
    """Class to save and load models using pickle."""
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(script_dir, "..", "..", "models")

    def save_model(self, model, file_name: str):
        """Saves the model to a file."""
        file_path = os.path.join(self.path, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_path}")

    def load_model(self, file_name: str):
        """Loads the model from a file."""
        file_path = os.path.join(self.path, file_name)
        if file_name not in os.listdir(self.path):
            raise FileNotFoundError(f"Model file '{file_name}' not found in {self.path}.")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return model
