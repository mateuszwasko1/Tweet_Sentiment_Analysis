from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from tweet_sentiment_analysis.deployment.process_deployment import PredictEmotion
from starlette.responses import RedirectResponse
from typing import List, Optional

app = FastAPI(
    title="Sentiment Analysis API",
    summary="Classify Tweet emotions via baseline or RoBERTa models",
    description="""
    ## API Usage

    Choose `?baseline=1` for the Logistic Regression baseline or omit /
    set to 0 for RoBERTa.

    ### Input format
    - Send a list of `{ text: string }` objects.

    ### Output format
    - Returns list of `{ input, prediction, confidence }`.
    """
)

class Input(BaseModel):
    text: str

class Prediction(BaseModel):
    input: str
    prediction: str
    confidence: float

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/info")
async def info(
    baseline: Optional[int] = Query(
        0,
        ge=0,
        le=1,
        description="1 for baseline, 0 for RoBERTa"
    )
):
    """Return model name and supported classes."""
    predictor = PredictEmotion(baseline=bool(baseline))
    model_name = "Baseline Logistic Regression" if (
        predictor.baseline) else "RoBERTa"
    return {
        "model": model_name,
        "classes": ["anger", "joy", "sadness", "fear"]
    }

@app.post("/predict")
async def predict(
    input_data: List[Input] = Body(...),
    baseline: Optional[int] = Query(
        0,
        ge=0,
        le=1,
        description="1 for baseline, 0 for RoBERTa"
    )
):
    """
    Predict emotion(s) for tweet(s). Use `?baseline=1` to select the
    Logistic Regression model.
    """
    predictor = PredictEmotion(baseline=bool(baseline))
    results: List[Prediction] = []
    for item in input_data:
        try:
            emotion, confidence = predictor.output_emotion(item.text)
        except Exception:
            raise HTTPException(status_code=422, detail="Prediction error")
        results.append(
            Prediction(input=item.text, prediction=emotion, confidence=confidence)
        )
    return results