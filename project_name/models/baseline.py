import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from project_name.models.save_load_model import ModelSaver
from project_name.preprocessing.baseline_preprocessing import (
    BaselinePreprocessor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV, PredefinedSplit

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


class BaselineModel:
    """
    Encapsulates training, evaluation, and plotting utilities for
    a logistic regression baseline classifier with TF-IDF features.
    """

    def __init__(self) -> None:
        """
        Initialize placeholders for model, data, and parameters.
        """
        self.best_parameters: dict[str, object] = {}
        self.preprocessor = BaselinePreprocessor()
        self.model: GridSearchCV | None = None
        self.X_test: scipy.sparse.spmatrix | None = None
        self.y_test: np.ndarray | None = None

    def train(
        self,
        data: tuple[
            tuple[scipy.sparse.spmatrix, np.ndarray],
            tuple[scipy.sparse.spmatrix, np.ndarray],
            tuple[scipy.sparse.spmatrix, np.ndarray],
        ],
    ) -> None:
        """
        Train a LogisticRegression using GridSearchCV over predefined folds,
        then save the best estimator.

        Args:
            data: A triple of (X_train, y_train), (X_dev, y_dev),
            (X_test, y_test) where X_* are sparse feature matrices
            and y_* are label arrays.
        """
        (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = data
        self.X_test = X_test
        self.y_test = y_test

        X_combined = scipy.sparse.vstack([X_train, X_dev])
        y_combined = np.concatenate([y_train, y_dev])
        train_fold = [-1] * X_train.shape[0] + [0] * X_dev.shape[0]
        ps = PredefinedSplit(test_fold=train_fold)

        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 20],
            "penalty": ["l1", "l2"],
            "solver": ["lbfgs", "saga"],
            "max_iter": [500, 1000, 2000],
            "class_weight": [None, "balanced"],
        }

        grid = GridSearchCV(
            estimator=LogisticRegression(),
            param_grid=param_grid,
            cv=ps,
            return_train_score=True,
            n_jobs=-1,
        )
        grid.fit(X_combined, y_combined)
        self.model = grid

        saver = ModelSaver()
        saver.save_model(model=grid, file_name="baseline_model")

    def predict(self) -> np.ndarray:
        """
        Predict labels for the held-out test set stored in self.X_test.

        Returns:
            Array of predicted labels for X_test.
        """
        assert self.model is not None, (
            "Model must be trained or loaded before prediction."
        )
        assert self.X_test is not None, "X_test is not set."
        preds = self.model.predict(self.X_test)
        self.best_parameters = self.model.best_params_
        return preds

    def evaluate(self) -> tuple[dict[str, object], np.ndarray]:
        """
        Generate classification report on test set predictions.

        Returns:
            A tuple of (best_parameters, predicted_labels).
        """
        preds = self.predict()
        assert self.y_test is not None, "y_test is not set."
        print(classification_report(self.y_test, preds))
        return self.best_parameters, preds

    def loss_plotter(self) -> None:
        """
        Plot training and validation scores against regularization C values.
        """
        assert self.model is not None, "Model must be trained before plotting."
        cv_results = self.model.cv_results_
        C_values = [params["C"] for params in cv_results["params"]]
        train_scores = cv_results["mean_train_score"]
        val_scores = cv_results["mean_test_score"]

        plt.figure(figsize=(8, 5))
        plt.plot(C_values, train_scores, marker="o", label="Train Score")
        plt.plot(C_values, val_scores, marker="o", label="Validation Score")
        plt.xscale("log")
        plt.xlabel("C (log scale)")
        plt.ylabel("Score")
        plt.title("Train vs Validation Score by C")
        plt.legend()
        plt.show()

    def plot_final_roc_curve(self) -> None:
        """
        Compute and display ROC curves for each class on the test set.
        """
        assert self.model is not None, (
            "Model must be trained before plotting ROC."
        )
        assert self.X_test is not None and self.y_test is not None, (
            "Test data not set."
        )

        y_score = self.model.predict_proba(self.X_test)
        classes = sorted(np.unique(self.y_test))
        y_test_bin = label_binarize(self.y_test, classes=classes)

        plt.figure(figsize=(10, 7))
        for idx, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_score[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.2f})"
            )

        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Baseline Model")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def pipeline(self, training: bool = True) -> None:
        """
        Run end-to-end: preprocess data, train or load model, then evaluate
        and plot performance metrics.

        Args:
            training: If True, retrain model; otherwise load existing model.
        """
        data = self.preprocessor.preprocessing_pipeline(at_inference=False)
        if training:
            self.train(data)
        else:
            (_, _), (_, _), (X_test, y_test) = data
            self.X_test = X_test
            self.y_test = y_test
            loader = ModelSaver()
            self.model = loader.load_model("baseline_model")

        self.evaluate()
        self.plot_final_roc_curve()
        self.loss_plotter()


if __name__ == "__main__":
    baseline = BaselineModel()
    baseline.pipeline(training=True)
