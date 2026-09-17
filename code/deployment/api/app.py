from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


#Path to the trained model
model_path = Path("/app/model/wine_model.joblib")


#Load model

#Load the trained Extra Trees model
model = joblib.load(model_path)


#Create FastAPI application

app = FastAPI(
    title="Wine Quality Prediction API",
    version="1.0.0"
)


#Input data structure

class WineData(BaseModel):

    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


#Health check

@app.get("/")
def root_endpoint():

    return {
        "message": "Wine Quality Prediction API is running"
    }


#Prediction endpoint

@app.post("/predict")
def predict(data: WineData):

    #Convert input data into the format expected by the model
    features = [[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]]

    #Make prediction
    prediction = model.predict(features)[0]

    return {
        "prediction": float(prediction)
    }