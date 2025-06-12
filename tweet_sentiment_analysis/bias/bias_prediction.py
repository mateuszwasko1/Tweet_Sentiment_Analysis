import pandas as pd
import os
from scipy.stats import binomtest
from typing import Tuple, Dict
from tweet_sentiment_analysis.bias.predictor_baseline import PredictBaseline
from tweet_sentiment_analysis.bias.prediction_bert import PredictBert


def bias_prediction(
        baseline: bool = False,
        bias_path: str = "data/bias_data/crows_pairs.csv"
        ) -> Tuple[float, Dict[str, float], float]:
    """
    Predicts bias on a given predictor and saves the results as a csv file.

    Args:
        predictor: Model predictor object that implements .predict()
        bias_path: path to the dataset
        save_path: path to the saved csv file

    Returns:
        mismatch_percent (float): Overall mismatch percentage
        mismatch_by_bias (Dict[str, float]): Mismatch percentages for each
        bias type.
        binomial_p_value (float): p-value of the binomial test on total
        mismatches.
    """
    if baseline:
        predictor = PredictBaseline()
    else:
        predictor = PredictBert()
    bias_df = pd.read_csv(bias_path)
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

    mismatches = (
        comparison_df[comparison_df["more_predict"] !=
                      comparison_df["less_predict"]]
                )
    mismatch_percent = (len(mismatches) / len(comparison_df)) * 100
    print(f"mismatched percentage: {mismatch_percent:.2f}%")

    total_per_bias = comparison_df.groupby("bias_type").size()
    mismatch_counts = mismatches.groupby("bias_type").size()
    mismatch_by_bias = (mismatch_counts / total_per_bias) * 100

    bias_threshold = 0.05

    binomial_test = binomtest(len(mismatches), len(comparison_df),
                              bias_threshold, alternative="greater")

    for bias_type, group in comparison_df.groupby("bias_type"):
        k = (group["more_predict"] != group["less_predict"]).sum()
        binomial_test_per_bias = binomtest(k, len(group), bias_threshold,
                                           alternative="greater")
        print(f"bias type: {bias_type}")
        print(f"Samples: {len(group)}")
        print(f"Mismatches:{k}")
        print(f"p-value: {binomial_test_per_bias.pvalue:.5f}")

    print("binomial test:", binomial_test)

    print(f"prediction mismatches per bias type: {mismatch_by_bias}")
    model_type = "baseline" if baseline else "bert"

    path = f"output/bias_results/{model_type}_bias_comparison.csv"
    path_mismatch = "output/bias_results/mismatch.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(path_mismatch), exist_ok=True)
    comparison_df.to_csv(path)
    mismatches.to_csv(path_mismatch)

    return mismatch_percent, mismatch_by_bias, binomial_test.pvalue
