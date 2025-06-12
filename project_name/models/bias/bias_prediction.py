import pandas as pd
import sys
import os
from scipy.stats import binomtest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from project_name.models.prediction_bert_ekphrasis import PredictEkphrasisBert

bias_path = "/kaggle/input/bias012345678901234/biasprediction/data/bias_data/crows_pairs.csv"

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

bias_threshold = 0.05

binomial_test = binomtest(len(mismatches), len(comparison_df), bias_threshold, alternative="greater")

for bias_type, group in comparison_df.groupby("bias_type"):
    k = (group["more_predict"] != group["less_predict"]).sum()
    binomial_test_per_bias = binomtest(k, len(group), bias_threshold, alternative="greater")
    print(f"bias type: {bias_type}")
    print(f"Samples: {len(group)}")
    print(f"Mismatches:{k}")
    print(f"p-value: {binomial_test_per_bias.pvalue:.5f}")

print("binomial test:", binomial_test)

print(f"prediction mismatches per bias type: {mismatch_by_bias}")
os.makedirs("/kaggle/working/bias_data", exist_ok=True)
comparison_df.to_csv("/kaggle/working/bias_data/bert_bias_comparison.csv")
