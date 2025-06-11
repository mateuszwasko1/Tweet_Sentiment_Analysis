import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..')))

from project_name.deployment.process_deployment import PredictEmotion

bias_path = "/kaggle/input/bias2345678/biasprediction/data/bias_data/" \
            "crows_pairs.csv"

bias_df = pd.read_csv(bias_path)

predictor = PredictEmotion(baseline=False)
predictor_baseline = PredictEmotion(baseline=True)

more_results, more_confs = predictor.predict(bias_df["sent_more"].tolist())
less_results, less_confs = predictor.predict(bias_df["sent_less"].tolist())

more_results_base, more_confs_base = predictor_baseline.predict(bias_df["sent_more"].tolist())
less_results_base, less_confs_base = predictor_baseline.predict(bias_df["sent_less"].tolist())

comparison_df = pd.DataFrame({
    "sent_more": bias_df["sent_more"],
    "more_predict": more_results,
    "more_conf": more_confs,
    "sent_less": bias_df["sent_less"],
    "less_predict": less_results,
    "less_conf": less_confs,
    "bias_type": bias_df["bias_type"]
})

comparison_df_baseline = pd.DataFrame({
    "sent_more_base": bias_df["sent_more"],
    "more_predict_base": more_results_base,
    "more_conf_base": more_confs_base,
    "sent_less_base": bias_df["sent_less"],
    "less_predict_base": less_results_base,
    "less_conf_base": less_confs_base,
    "bias_type_base": bias_df["bias_type"]
})

# use for binom test
comparison_df["conf_diff"] = (
    comparison_df["more_conf"] - comparison_df["less_conf"]).abs()

# use for binom test of baseline predictions
comparison_df["conf_diff_base"] = (
    comparison_df["more_conf_base"] - comparison_df["less_conf_base"]).abs()

# overall mismatch for BERT model
mismatches = comparison_df[comparison_df["more_predict"] != comparison_df[
    "less_predict"]]
mismatch_percent = (len(mismatches) / len(comparison_df)) * 100
print(f"mismatched percentage in BERT model: {mismatch_percent:.2f}%")

# overall mismatch for Baseline Model
mismatches_baseline = (
    comparison_df[comparison_df["more_predict_base"] != comparison_df[
        "less_predict_base"]])
mismatch_percent_baseline = (
    (len(mismatches_baseline) / len(comparison_df_baseline)) * 100)
print(
    f"mismatched percentage in Baseline Model: "
    f"{mismatch_percent_baseline:.2f}%"
    )


# mismatch per group per in BERT Model
total_per_bias = comparison_df.groupby("bias_type").size()
mismatch_counts = mismatches.groupby("bias_type").size()
mismatch_by_bias = (mismatch_counts / total_per_bias) * 100

# mismatch per group per in Baseline Model
total_per_bias_base = comparison_df_baseline.groupby("bias_type").size()
mismatch_counts_base = mismatches_baseline.groupby("bias_type").size()
mismatch_by_bias_baseline = (mismatch_counts_base / total_per_bias_base) * 100

print(f"prediction mismatches per bias type for BERT: {mismatch_by_bias}")
print(
    f"prediction mismatches per bias type for Baseline: "
    f"{mismatch_by_bias_baseline}"
    )
os.makedirs("/kaggle/working/bias_data", exist_ok=True)
mismatches.to_csv("/kaggle/working/bias_data/bert_bias_comparison.csv")
mismatches_baseline.to_csv(
    "/kaggle/working/bias_data/bert_bias_comparison_baseline.csv")
