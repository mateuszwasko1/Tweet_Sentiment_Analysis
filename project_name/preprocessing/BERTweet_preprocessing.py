import pandas as pd


class BERTweetPreprocessor():
    # def __init__(self):
    #     pass

    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str
                                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Add value error to check if allowed names are passed on
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def encode_labels(self, df):
        emotion_map = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}
        df['emotion'] = df['emotion'].map(emotion_map)
        if df['emotion'].isnull().any():
            raise ValueError("DataFrame contains invalid emotion labels.")
        return df

    def preprocess_df(self, df: pd.DataFrame):
        df = self.encode_labels(df)
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        return X, y
