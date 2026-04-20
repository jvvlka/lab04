from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = FastAPI()

APP_NAME = os.getenv("APP_NAME", "ML API default")

model = LinearRegression()
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])
model.fit(X_train, y_train)

class PredictionInput(BaseModel):
    feature: float

@app.get("/")
def read_root():
    return {"message": "API", "app_name": APP_NAME}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array([[data.feature]])
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail="brak wymaganej wartości lub nieprawidłowy format danych.")

@app.get("/info")
def get_info():
    return {"model_type": "LinearRegression", "n_features": 1}

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.get("/config")
def get_config():
    return {"APP_NAME": APP_NAME}
