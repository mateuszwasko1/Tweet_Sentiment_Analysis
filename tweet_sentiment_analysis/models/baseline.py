from tweet_sentiment_analysis.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
from tweet_sentiment_analysis.models.save_load_model import ModelSaver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                             '..')))


class BaselineModel():
    """
    Encapsulates training, evaluation, and plotting utilities for
    a logistic regression baseline classifier with TF-IDF features.
    """

    def __init__(self) -> None:
        """
        Initialize placeholders for model, data, and parameters.
        """
        self.best_parameters = None
        self.preprocessor = BaselinePreprocessor()
        self.model = None
        self.X_test = None
        self.y_test = None

    def train(self, data: tuple[
            tuple[scipy.sparse.spmatrix, np.ndarray],
            tuple[scipy.sparse.spmatrix, np.ndarray],
            tuple[scipy.sparse.spmatrix, np.ndarray],
            ]
             ) -> None:
        """
        Trains a logistic regression model using GridSearchCV, saves the
        fitted grid to the class instance, and saves the best model using
        ModelSaver in data/models.

        Args:
            data (tuple): A triple tuple containing three pairs of (X, y) for
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
                            return_train_score=True, n_jobs=-1)
        grid.fit(X_combined, y_combined)
        self.model = grid
        model_saver = ModelSaver()
        model_saver.save_model(model=grid, file_name="baseline_model")

    def predict(self) -> np.ndarray:
        """
        Predict labels for the held-out test set stored in self.X_test.

        Returns:
            Array of predicted labels for X_test.
        """
        assert self.model is not None, (
            "Model must be trained or loaded before prediction."
        )
        grid_predictions = self.model.predict(self.X_test)
        self.best_parameters = self.model.best_params_
        return grid_predictions

    def evaluate(self) -> tuple[dict[str, object], np.ndarray]:
        """
        Evaluates the model by loading it if not already loaded,
        predicting on the test set, printing the classification report,
        and plotting the loss curve.
        """
        grid_predictions = self.predict()
        print(classification_report(self.y_test, grid_predictions))
        return grid_predictions

    def plot_final_roc_curve(self) -> None:
        """
        Plots ROC curves for the best model on the test set.
        """
        y_score = self.model.predict_proba(self.X_test)

        classes = sorted(np.unique(self.y_test))
        y_test_bin = label_binarize(self.y_test, classes=classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, class_id in enumerate(classes):
            fpr[class_id], tpr[class_id], _ = roc_curve(
                y_test_bin[:, i], y_score[:, i])
            roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

        plt.figure(figsize=(10, 7))
        for class_id in classes:
            plt.plot(fpr[class_id], tpr[class_id], label=(
                f"Class {class_id} (AUC = {roc_auc[class_id]:.2f})"))

        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Baseline Model")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, y_pred: np.ndarray) -> None:
        """
        Plots the confusion matrix, shows it, and prints the raw table/
        """
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Baseline Confusion Matrix")
        plt.show()
        plt.close()

    def pipeline(self, training: bool = True) -> None:
        """
        Run end-to-end: preprocess data, train or load model, then evaluate
        and plot performance metrics.

        Args:
            training: If True, retrain model; otherwise load existing model.
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
        predictions = self.evaluate()
        self.plot_final_roc_curve()
        self.plot_confusion_matrix(predictions)
