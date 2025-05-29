from project_name.models.bert import BertModel
from project_name.models.baseline import BaselineModel

if __name__ == '__main__':
    type_of_model = "Bert"
    if type_of_model == "Baseline":
        baseline = BaselineModel()
        print(baseline.pipeline())
    elif type_of_model == "Bert":
        model = BertModel()
        model.pipeline()
        
