from fastapi import FastAPI
from app.schemas import CarInput
from app.utils import predict_price

app = FastAPI(title="Car Price Prediction API")

@app.get("/")
def home():
    return {"message": "Car Price Prediction API is running"}

@app.post("/predict")
def predict(car: CarInput):
    price = predict_price(car.dict())
    return {"predicted_price": price}
