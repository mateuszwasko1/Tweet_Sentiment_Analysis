from typing import List, Dict

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from project_name.deployment.process_deployment import PredictEmotion

predictor: PredictEmotion = PredictEmotion(baseline=False)

app: FastAPI = FastAPI(
    title="Logistic Regression Sentiment Analysis",
    summary="Logistic regression–based tweet emotion API",
    description="""
    ## API Usage

    This API classifies the emotion of tweets using a
    trained Logistic Regression model.

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

    - The API returns a list of objects, each with the original
    input and the predicted emotion.

    #### Example response:

    ```json
    [
      {
        "input": "I love you",
        "prediction": "joy",
        "confidence": 0.95
      },
      {
        "input": "I hate you",
        "prediction": "anger",
        "confidence": 0.92
      }
    ]
    ```
    """,
)


class Input(BaseModel):
    """
    Schema for a single tweet input.

    Attributes:
        text (str): The raw tweet text to classify.
    """
    text: str


class Prediction(BaseModel):
    """
    Schema for a single prediction output.

    Attributes:
        input (str): The original tweet text.
        prediction (str): The predicted emotion label.
        confidence (float): Confidence score of the prediction.
    """
    input: str
    prediction: str
    confidence: float


@app.get("/", response_model=None)
async def root() -> RedirectResponse:
    """
    Redirect the root endpoint to the API documentation.

    Returns:
        RedirectResponse: Redirects to '/docs'.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=Dict[str, str])
async def health() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict[str, str]: Status message indicating API health.
    """
    return {"status": "ok"}


@app.get("/info", response_model=Dict[str, List[str]])
async def info() -> Dict[str, List[str]]:
    """
    Provide model name and supported classes.

    Returns:
        Dict[str, List[str]]: Model name under 'model'
        key and list of classes under 'classes'.
    """
    model_name = (
        "Baseline Logistic Regression" if predictor.baseline else "RoBERTa"
    )
    return {
        "model": model_name, "classes": ["anger", "joy", "sadness", "fear"]
    }


@app.post("/predict", response_model=List[Prediction])
async def predict(
    input_data: List[Input] = Body(
        ..., description="List of tweet objects to classify"
    )
) -> List[Prediction]:
    """
    Predict emotion(s) for one or more tweets.

    Args:
        input_data (List[Input]): List of tweet input objects.

    Returns:
        List[Prediction]: Prediction results with input,
        predicted label, and confidence.

    Raises:
        HTTPException: If prediction fails due to invalid
        input or processing error.
    """
    results: List[Prediction] = []
    for item in input_data:
        try:
            label, confidence = predictor.output_emotion(item.text)
        except Exception:
            raise HTTPException(status_code=422, detail="validation error")
        results.append(Prediction(
            input=item.text, prediction=label, confidence=confidence
        ))
    return results
