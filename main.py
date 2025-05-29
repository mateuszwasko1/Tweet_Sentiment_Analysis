from project_name.models.baseline import BaselineModel
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from  project_name.deployment.deployment import predict_emotion
from starlette.responses import RedirectResponse
from typing import List, Union
import uvicorn

app = FastAPI(
    title="Bert Tweet Sentiment Analysis",
    summary="An API endpoint to classify emotions of Tweets using Bert "
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


@app.post("/predict")
async def predict(input_data: Union[Input, List[Input]] = Body(...)):
    if isinstance(input_data, Input):
        try:
            emotion = predict_emotion(input_data.text)
        except Exception:
            raise HTTPException(status_code=422, detail="validation error")
        return Prediction(input=input_data.text, prediction=emotion)
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
    print(baseline.pipeline())
"""
