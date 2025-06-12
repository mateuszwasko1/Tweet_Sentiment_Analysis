import unittest
import os
import json
from tweet_sentiment_analysis import BaselinePreprocessor
import pandas as pd
import pandas.testing as pd_t

class Preprocessing_Test(unittest.TestCase):
    def setUp(self):
        self.processor = BaselinePreprocessor()


    def test_extract_feature_lables(self):
        d = {'tweet': ["This is a angry tweet", "This is a sad tweet"], "emotion": ["angry", "sadness"]}
        df = pd.DataFrame(data=d)

        X, y = self.processor.extract_features_labels(df, "tweet", "emotion")

        actual_X = pd.Series(["This is a angry tweet", "This is a sad tweet"], name="tweet")
        pd_t.assert_series_equal(X, actual_X)

        actual_y = pd.Series(["angry","sadness"], name="emotion")
        pd_t.assert_series_equal(y, actual_y)

    def test_clean_text(self):
        text = "HELLO @world, how are you"
        text = self.processor.clean_text(text)

        actual_text = "hello user, how are you"
        self.assertEqual(text, actual_text)

    def test_vectorize(self):
        # Testing for test set
        X_train = pd.Series(["This is a angry tweet", "This is a sad tweet"], name="tweet")
        X_train_vec = self.processor.vectorize(X_train)
        # The shape should be 2,5 because we have 2 data points and the longest one is 5 words long
        self.assertEqual(X_train_vec.shape, (2,5))

        # Testing for test/dev set
        self.processor.training_data = False
        X_test = pd.Series(["This is a angry tweet in test data set"], name="tweet")
        self.assertEqual(X_test.shape, (1,))

        # Reset the parameters for the next test
        self.processor.training_data = True

    def test_preprocess_df(self):
        d = {'tweet': ["This is a angry tweet", "This is a sad tweet"], "emotion": ["angry", "sadness"]}
        df = pd.DataFrame(data=d)

        self.processor.preprocess_df(df)