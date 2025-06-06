import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))
from project_name.preprocessing.baseline_preprocessing import BaselinePreprocessor
from project_name.models.save_load_model import ModelSaver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt


class BaselineModel():
    def __init__(self):
        self.best_parameters = None
        self.preprocessor = BaselinePreprocessor()
        self.model = None
        self.X_test = None
        self.y_test = None

    def train(self, data):
        """
        Trains a logistic regression model using GridSearchCV, saves the
        fitted grid to the class instance, and saves the best model using
        ModelSaver in data/models.

        Args:
            data (tuple): A tuple containing three pairs of (X, y) for
            training, development, and test sets:
                ((X_training, y_training), (X_dev, y_dev), (X_test, y_test))
        """
        (X_training, y_training), (X_dev, y_dev), (X_test, y_test) = data
        self.X_test = X_test
        self.y_test = y_test
        X_combined = scipy.sparse.vstack([X_training, X_dev])
        y_combined = np.concatenate([y_training, y_dev])
        train_fold = [-1]*X_training.shape[0] + [0]*X_dev.shape[0]
        ps = PredefinedSplit(test_fold=train_fold)

        parameters = {
            "C": [0.001, 0.01, 0.1, 1, 10, 20],
            "penalty": ["l1", "l2"],
            "solver": ["lbfgs", "saga"],
            "max_iter": [500, 1000, 2000],
            "class_weight": [None, "balanced"]
        }

        grid = GridSearchCV(LogisticRegression(), parameters, cv=ps,
                            return_train_score=True)
        grid.fit(X_combined, y_combined)
        self.model = grid
        model_saver = ModelSaver()
        model_saver.save_model(model=grid, file_name="baseline_model")

    def predict(self):
        grid_predictions = self.model.predict(self.X_test)
        self.best_parameters = self.model.best_params_
        return grid_predictions

    def evaluate(self):
        """
        Evaluates the model by loading it if not already loaded,
        predicting on the test set, printing the classification report,
        and plotting the loss curve.
        """
        grid_predictions = self.predict()
        print(classification_report(self.y_test, grid_predictions))
        self.loss_plotter()
        return self.best_parameters, grid_predictions

    def loss_plotter(self):
        grid = self.model
        mean_train_scores = grid.cv_results_['mean_train_score']
        mean_val_scores = grid.cv_results_['mean_test_score']
        param_C = [params['C'] for params in grid.cv_results_['params']]

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(param_C, mean_train_scores, label='Training Score',
                 marker='o')
        plt.plot(param_C, mean_val_scores, label='Validation Score',
                 marker='o')
        plt.xscale('log')
        plt.xlabel('C (log scale)')
        plt.ylabel('Score')
        plt.title('Training vs Validation Score for Different C')
        plt.legend()
        plt.show()

    def pipeline(self, training=True):
        """
        Runs the preprocessing pipeline, trains the model, prints the
        classification report for the grid predictions on the test set,
        and displays the model's performance.
        """
        preprocesser_tfidf = BaselinePreprocessor()
        data = preprocesser_tfidf.preprocessing_pipeline(at_inference=False)
        if training is True:
            self.train(data)
        else:
            (_, _), (_, _), (X_test, y_test) = data
            self.X_test = X_test
            self.y_test = y_test
            model_loader = ModelSaver()
            self.model = model_loader.load_model("baseline_model")
        self.evaluate()
        self.loss_plotter()


if __name__ == "__main__":
    baseline_model = BaselineModel()
    baseline_model.pipeline(training=True)
