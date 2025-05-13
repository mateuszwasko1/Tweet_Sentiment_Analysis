from preprocessing.baseline_preprocessing import BaselinePreprocessor


class BaselineModel:
    def __init__(self):
        pass

    def pipeline(self):
        # do preprocessing and run model, and evaluate
        preprocesser = BaselinePreprocessor()
        preprocesser.preprocessing_pipeline()
        pass
