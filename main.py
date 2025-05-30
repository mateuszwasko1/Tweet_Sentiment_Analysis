from project_name.models.bert_ekphrasis import BertModel
from project_name.models.baseline import BaselineModel
from project_name.models.prediction_bert_ekphrasis import PredictEkphrasisBert

if __name__ == '__main__':
    type_of_model = "Bert_p"
    if type_of_model == "Baseline":
        baseline = BaselineModel()
        print(baseline.pipeline())
    elif type_of_model == "Bert":
        model = BertModel()
        model.pipeline()
    elif type_of_model == "Bert_p":
        prediction = PredictEkphrasisBert()
        while 1==1:
            text = input("What text would you like predict?")
            label_class, prob = prediction.predict(text)
            print(f"The predicted class is {label_class} with a probability of {(prob*100):.2f}%.")
