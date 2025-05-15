from project_name.preprocessing import BaselinePreprocessor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score


class BaselineModel:

    def regression(self, regression_type):
        if regression_type == "logisitc":
            regression = LogisticRegression(max_iter=1000)
        else:
            regression = LinearRegression()

        regression.fit(self.X_training, self.y_training)
        y_prediction = regression.predict(self.X_test)

        if regression_type == "logisitc":
            return classification_report(self.y_test, y_prediction)
        else:
            mse = mean_squared_error(self.y_test, y_prediction)
            r2 = r2_score(self.y_test, y_prediction)
            return mse, r2

    def pipeline(self):
        preprocesser_tfidf = BaselinePreprocessor(use_tfidf=True)
        (self.X_training, self.y_training), (self.X_test, self.y_test) =\
            preprocesser_tfidf.preprocessing_pipeline()

        logistic_results = self.regression(regression_type="logisitc")
        print(logistic_results)


        preprocesser_glove = BaselinePreprocessor(use_tfidf=False)
        (self.X_training, self.y_training), (self.X_test, self.y_test) =\
            preprocesser_glove.preprocessing_pipeline()
        linear_results = self.regression(regression_type="linear")
        print(linear_results)

if __name__ == "__main__":
    baseline = BaselineModel()
    baseline.pipeline()
