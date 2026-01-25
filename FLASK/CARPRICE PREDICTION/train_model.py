import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\FLASK PROJECTS\CAR PRICE PREDICTION\data\car_prediction_data.csv")

# ❌ DROP Car_Name
df = df.drop(columns=["Car_Name"])

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

num_features = ["Year", "Present_Price", "Kms_Driven", "Owner"]
cat_features = ["Fuel_Type", "Seller_Type", "Transmission"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# ✅ SAVE MODEL (CORRECT PATH)
joblib.dump(
    pipeline,
    r"C:\Users\LENOVO\Downloads\FLASK PROJECTS\CAR PRICE PREDICTION\model\car_price_model.pkl"
)
