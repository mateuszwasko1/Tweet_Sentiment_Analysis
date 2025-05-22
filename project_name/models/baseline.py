from project_name.preprocessing import BaselinePreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class BaselineModel:

    def regression(self, data):
        (X_training, y_training), (X_test, y_test) = data

        parameters = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [1000]
        }

        grid = GridSearchCV(LogisticRegression(), parameters, cv = 5)
        grid.fit(X_training, y_training)

        print(grid.best_params_)

        grid_predictions = grid.predict(X_test)
        return classification_report(y_test, grid_predictions)

        # regression = LogisticRegression(max_iter=1000)
        # regression.fit(X_training, y_training)
        # y_prediction = regression.predict(X_test)
        # return classification_report(y_test, y_prediction)

    def pipeline(self):
        preprocesser_tfidf = BaselinePreprocessor()
        data = preprocesser_tfidf.preprocessing_pipeline()

        return self.regression(data)
