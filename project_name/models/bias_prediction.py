import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from project_name.models.prediction_bert_ekphrasis import PredictEkphrasisBert

bias_path = "/kaggle/input/bias2345678/biasprediction/data/bias_data/crows_pairs.csv"

bias_df = pd.read_csv(bias_path)

predictor = PredictEkphrasisBert()

more_results, more_confs = predictor.predict(bias_df["sent_more"].tolist())
less_results, less_confs = predictor.predict(bias_df["sent_less"].tolist())

comparison_df = pd.DataFrame({
    "sent_more": bias_df["sent_more"],
    "more_predict": more_results,
    "more_conf": more_confs,
    "sent_less": bias_df["sent_less"],
    "less_predict": less_results,
    "less_conf": less_confs,
    "bias_type": bias_df["bias_type"]
})

comparison_df["conf_diff"] = (
    comparison_df["more_conf"] - comparison_df["less_conf"]).abs()

mismatches = comparison_df[comparison_df["more_predict"] != comparison_df["less_predict"]]
mismatch_percent = (len(mismatches) / len(comparison_df)) * 100
print(f"mismatched percentage: {mismatch_percent:.2f}%")

total_per_bias = comparison_df.groupby("bias_type").size()
mismatch_counts = mismatches.groupby("bias_type").size()
mismatch_by_bias = (mismatch_counts / total_per_bias) * 100

print(f"prediction mismatches per bias type: {mismatch_by_bias}")
os.makedirs("/kaggle/working/bias_data", exist_ok=True)
mismatches.to_csv("/kaggle/working/bias_data/bert_bias_comparison.csv")
