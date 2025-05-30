from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from project_name.deployment.deployment import predict_emotion
from starlette.responses import RedirectResponse
from typing import List

app = FastAPI(
    title="Logistic Regression Sentiment Analysis",
    summary="An API endpoint to classify emotions of Tweets using Logistic Regression",
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


"""
if __name__ == '__main__':
    baseline = BaselineModel()
    baseline_metrics =baseline.pipeline()
    print(baseline.best_parameters)
    print(baseline_metrics)

