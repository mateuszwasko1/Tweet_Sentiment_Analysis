from tweet_sentiment_analysis.models.bert_model import BertModel
from tweet_sentiment_analysis.models.baseline import BaselineModel
from tweet_sentiment_analysis.bias.bias_prediction import bias_prediction
import os

if __name__ == '__main__':
    if os.path.exists("models/saved_bert/model/model.safetensors"):
        decision = input(
            "Would you like to train the model "
            "or evaluate bias (model/bias)").lower()
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
