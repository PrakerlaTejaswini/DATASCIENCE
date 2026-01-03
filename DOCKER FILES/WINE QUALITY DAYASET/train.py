import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\LENOVO\Documents\Wine Quality Dataset by Docker\data\winequality-red.csv")

# Binary classification:
# Good wine (quality >= 7) → 1
# Bad wine (quality < 7) → 0
df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model.pkl")
print("Model saved as model.pkl")
