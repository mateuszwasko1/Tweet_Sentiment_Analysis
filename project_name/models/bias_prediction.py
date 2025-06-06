import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from project_name.models.prediction_bert_ekphrasis import PredictEkphrasisBert

male_df = pd.read_csv("data/bias_data/eval_male.csv")
female_df = pd.read_csv("data/bias_data/eval_female.csv")

predictor = PredictEkphrasisBert()

male_results = male_df["Text"].apply(predictor.predict)
female_results = female_df["Text"].apply(predictor.predict)

comparison_df = pd.DataFrame({
    "text_male": male_df["Text"],
    "male_predict": male_results.apply(lambda x: x[0]),
    "male_conf": male_results.apply(lambda x: x[1]),
    "text_female": female_df["Text"],
    "female_predict": female_results.apply(lambda x: x[0]),
    "female_conf": female_results.apply(lambda x: x[1]),
})

comparison_df.to_csv("data/bias_data/bert_bias_comparison.csv")
print(comparison_df.head())
