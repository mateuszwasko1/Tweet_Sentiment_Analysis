from project_name.models.baseline import BaselineModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from  project_name.deployment.deployment import predict_emotion
from starlette.responses import RedirectResponse

app = FastAPI(
    title="Bert Tweet Sentiment Analysis",
    summary="An API endpoint to classify emotions of Tweets using Bert "
)


class Input(BaseModel):
    text: str


class Prediction(BaseModel):
    input: str
    prediction: str


@app.get("/")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/predict", response_model=Prediction)
async def predict(input: Input):
    try:
        emotion = predict_emotion(input.text)
    except Exception: 
        raise HTTPException(status_code=422, detail="validation error")
    return Prediction(input=input.text, prediction=emotion)


"""
if __name__ == '__main__':
    baseline = BaselineModel()
    print(baseline.pipeline())
"""
