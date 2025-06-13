from tweet_sentiment_analysis.models.bert_model import BertModel
from tweet_sentiment_analysis.models.baseline import BaselineModel
from tweet_sentiment_analysis.bias.bias_prediction import bias_prediction
import os

"""
By running this file you can either train the baseline model (logistic
regression) or roBERTa. If there is a model in the directory you can run bias
analysis on it.

To run ensure you are in the environment by running "pipenv shell" and then
"pipenv install" and then run the file "python training_or_bias.py".
"""
if __name__ == '__main__':
    if os.path.exists(
        "output/saved_bert/model/model.safetensors") or os.path.exists(
            "output/baseline/baseline_model"):
        decision = input(
            "Would you like to train the model "
            "or evaluate bias (model/bias):").lower()
        bias = False if decision == "model" else True
        action = input("Which model do you want to use "
                       "for your selected task (logistic/roberta):").lower()
    else:
        bias = False
        action = input(
            "What model are you training (logistic/roberta):").lower()

    if not bias and action == "logistic":
        baseline = BaselineModel()
        print(baseline.pipeline())
    elif not bias and action == "roberta":
        model = BertModel()
        model.pipeline()
    elif bias and action == "logistic":
        bais = bias_prediction(baseline=True)
    elif bias and action == "roberta":
        bais = bias_prediction(baseline=False)
    else:
        print("Something went wrong. Check spelling or"
              "if the model is corrupt")
