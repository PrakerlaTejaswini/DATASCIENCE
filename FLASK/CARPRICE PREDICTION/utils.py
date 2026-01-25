import pandas as pd
import joblib

MODEL_PATH = r"C:\Users\LENOVO\Downloads\FLASK PROJECTS\CAR PRICE PREDICTION\model\car_price_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_price(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)
