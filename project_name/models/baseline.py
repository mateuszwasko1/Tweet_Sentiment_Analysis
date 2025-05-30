from project_name.preprocessing import BaselinePreprocessor
from project_name.models.save_load_model import ModelSaver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import numpy as np
import scipy.sparse


class BaselineModel:

    def regression(self, data):
        (X_training, y_training), (X_dev, y_dev), (X_test, y_test) = data

        X_combined = scipy.sparse.vstack([X_training, X_dev])
        y_combined = np.concatenate([y_training,y_dev])
        train_fold = [-1]*X_training.shape[0] + [0]*X_dev.shape[0]
        ps = PredefinedSplit(test_fold=train_fold)

        parameters = {
            "C": [0.001, 0.01, 0.1, 1, 10, 20],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [500, 1000, 2000],
            "class_weight": [None, "balanced"]
        }

        grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
        grid.fit(X_training, y_training)
        print(grid.best_params_)
        model_saver = ModelSaver()
        model_saver.save_model(model=grid, file_name="baseline_model")
        grid_predictions = grid.predict(X_test)
        self.best_parameters = grid.best_params_
        return classification_report(y_test, grid_predictions)

    def pipeline(self):
        preprocesser_tfidf = BaselinePreprocessor()
        data = preprocesser_tfidf.preprocessing_pipeline(at_inference=False)

        return self.regression(data)
