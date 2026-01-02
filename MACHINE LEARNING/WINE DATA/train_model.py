import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load CSV (IMPORTANT: comma-separated)
df = pd.read_csv(r"C:\Users\LENOVO\Documents\AIML PROJECTS\WINE PREDICTION\winequality-red.csv", sep=",")

# Verify columns
print("Columns loaded:")
print(df.columns)

# Create quality labels
def quality_label(q):
    if q <= 4:
        return "Low"
    elif q <= 6:
        return "Medium"
    else:
        return "High"

df["quality_label"] = df["quality"].apply(quality_label)

# Features & target
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
with open("wine_model.pkl", "wb") as f:
    pickle.dump({
        "model": pipeline,
        "features": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": accuracy,
        "dataframe": df
    }, f)

print("✅ Training completed successfully")
print("✅ Accuracy:", accuracy)
