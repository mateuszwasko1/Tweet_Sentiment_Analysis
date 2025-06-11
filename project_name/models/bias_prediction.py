"""
bias_prediction.py

Load sentence pairs with different bias levels, run emotion predictions,
and analyze mismatches between the 'more' and 'less' variants.
"""

import os
import sys

import pandas as pd
from pandas import DataFrame

from project_name.deployment.process_deployment import PredictEmotion

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)

bias_path: str = (
    "/kaggle/input/bias2345678/"
    "biasprediction/data/bias_data/"
    "crows_pairs.csv"
)

bias_df: DataFrame = pd.read_csv(bias_path)

predictor: PredictEmotion = PredictEmotion(baseline=False)

more_results: list[str]
more_confs: list[float]
more_results, more_confs = predictor.predict(
    bias_df["sent_more"].tolist()
)

less_results: list[str]
less_confs: list[float]
less_results, less_confs = predictor.predict(
    bias_df["sent_less"].tolist()
)

comparison_df: DataFrame = pd.DataFrame(
    {
        "sent_more": bias_df["sent_more"],
        "more_predict": more_results,
        "more_conf": more_confs,
        "sent_less": bias_df["sent_less"],
        "less_predict": less_results,
        "less_conf": less_confs,
        "bias_type": bias_df["bias_type"],
    }
)

comparison_df["conf_diff"] = (
    comparison_df["more_conf"]
    - comparison_df["less_conf"]
).abs()

mismatches: DataFrame = comparison_df[
    comparison_df["more_predict"] != comparison_df["less_predict"]
]

mismatch_percent: float = len(mismatches) / len(comparison_df) * 100.0
print(f"mismatched percentage: {mismatch_percent:.2f}%")

total_per_bias: pd.Series = comparison_df.groupby("bias_type").size()
mismatch_counts: pd.Series = mismatches.groupby("bias_type").size()
mismatch_by_bias: pd.Series = (
    mismatch_counts / total_per_bias * 100.0
)
print("prediction mismatches per bias type:")
print(mismatch_by_bias)

output_dir: str = "/kaggle/working/bias_data"
os.makedirs(output_dir, exist_ok=True)
mismatches.to_csv(
    os.path.join(output_dir, "bert_bias_comparison.csv"),
    index=False,
)
