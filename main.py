from project_name.models.bert_model import BertModel
from project_name.models.baseline import BaselineModel
from project_name.deployment.process_deployment import PredictEmotion

"""
This script selects and runs an emotion prediction model pipeline based on
the specified model type.

- Baseline: Runs BaselineModel.pipeline() and prints the result.
- Bert: Runs BertModel.pipeline().
- Bert_p: Interactively prompts user for predictions
using PredictEmotion.predict().
"""


def main() -> None:
    """
    Main entry point for running the selected model pipeline.

    Sets the `type_of_model` and executes the corresponding pipeline:
      * "Baseline" runs the baseline model and prints its output.
      * "Bert" runs the BERT model pipeline.
      * "Bert_p" prompts the user for multiple emotion predictions.
    """
    type_of_model: str = "Baseline"

    if type_of_model == "Baseline":
        baseline: BaselineModel = BaselineModel()
        result: str = baseline.pipeline()
        print(result)

    elif type_of_model == "Bert":
        model: BertModel = BertModel()
        model.pipeline()

    elif type_of_model == "Bert_p":
        predictor: PredictEmotion = PredictEmotion()

        number_of_predictions: int = int(
            input(
                "How many predictions would you like to make? "
            )
        )
        if number_of_predictions <= 0:
            raise ValueError("Number of predictions must be greater than 0.")

        count: int = 0
        while count < number_of_predictions:
            count += 1
            text: str = input("What text would you like to predict? ")
            label_class, prob = predictor.predict(text)
            label_class: str
            prob: float
            print(
                f"The predicted class is {label_class} with a probability of {
                    (prob * 100):.2f
                }%."
            )


if __name__ == "__main__":
    main()
