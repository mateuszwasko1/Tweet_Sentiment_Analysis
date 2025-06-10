from project_name.models.bert_model import BertModel
from project_name.models.baseline import BaselineModel
from project_name.deployment.process_deployment import PredictEmotion
'''
app = FastAPI(
    title="Logistic Regression Sentiment Analysis",
    summary="An API endpoint to classify emotions of Tweets using Logistic
    Regression",
    description="""
    ## API Usage

    This API classifies the emotion of tweets using a trained Logistic
    Regression model.

    ### **Input format**

    - **Always send a list of objects, even for a single tweet.**

    #### Example (single tweet):

    ```json
    [
    { "text": "I love you" }
    ]
    ```

    #### Example (multiple tweets):

    ```json
    [
    { "text": "I love you" },
    { "text": "I hate you" }
    ]
    ```

    ### **Output format**

    - The API returns a list of objects, each with the original input and the
    predicted emotion.

    #### Example response:

    ```json
    [
    {
        "input": "I love you",
        "prediction": "joy"
    },
    {
        "input": "I hate you",
        "prediction": "anger"
    }
    ]
    ```
    """
)


class Input(BaseModel):
    text: str


class InputList(BaseModel):
    inputs: List[Input]


class Prediction(BaseModel):
    input: str
    prediction: str


@app.get("/")
async def root():
    return RedirectResponse(url='/docs')


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/info")
async def info():
    return {
        "model": "LogisticRegression",
        "version": "1.0",
        "classes": ["anger", "joy", "sadness", "fear"]
    }


@app.post("/predict")
async def predict(input_data: List[Input] = Body(...)):
    """
    Predict emotion(s) for tweet(s).

    - Always send a list of objects:
      [{"text": "your tweet here"}]
    """

    results = []
    for item in input_data:
        try:
            emotion = predict_emotion(item.text)
        except Exception:
            raise HTTPException(status_code=422, detail="validation error")
        results.append(Prediction(input=item.text, prediction=emotion))
    return results


'''
if __name__ == '__main__':
    type_of_model = "Baseline"
    if type_of_model == "Baseline":
        baseline = BaselineModel()
        print(baseline.pipeline())
    elif type_of_model == "Bert":
        model = BertModel()
        model.pipeline()
    elif type_of_model == "Bert_p":
        prediction = PredictEmotion()
        number_of_predictions = int(input("How many predictions would you like\
        to make?"))
        if number_of_predictions <= 0:
            raise ValueError("Number of predictions must be greater than 0.")
        i = 0
        while i < number_of_predictions:
            i += 1
            text = input("What text would you like predict?")
            label_class, prob = prediction.predict(text)
            print(f"The predicted class is {label_class} with a probability of\
                  {(prob*100):.2f}%.")
            # print(prediction.predict(text))
