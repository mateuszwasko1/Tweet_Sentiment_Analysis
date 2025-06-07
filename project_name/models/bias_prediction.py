import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from project_name.models.prediction_bert_ekphrasis import PredictEkphrasisBert

male_df = pd.read_csv("data/bias_data/eval_male.csv")
female_df = pd.read_csv("data/bias_data/eval_female.csv")

predictor = PredictEkphrasisBert()

male_results, male_confs = predictor.predict(male_df["Text"].tolist())
female_results, female_confs = predictor.predict(female_df["Text"].tolist())

comparison_df = pd.DataFrame({
    "text_male": male_df["Text"],
    "male_predict": male_results,
    "male_conf": male_confs,
    "text_female": female_df["Text"],
    "female_predict": female_results,
    "female_conf": female_confs,
})

comparison_df.to_csv("data/bias_data/bert_bias_comparison.csv")
print(comparison_df.head())
