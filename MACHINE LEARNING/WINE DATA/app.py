import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load saved data
with open(r"C:\Users\LENOVO\Documents\AIML PROJECTS\wine_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
features = data["features"]
X_test = data["X_test"]
y_test = data["y_test"]
accuracy = data["accuracy"]
df = data["dataframe"]

st.title("🍷 Wine Quality Prediction – ML Project")

# ---------------- Model Performance ----------------
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

# ---------------- EDA: Correlation Heatmap ----------------
st.subheader("📈 Exploratory Data Analysis")
corr = df.drop("quality_label", axis=1).corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------- User Input ----------------
st.subheader("🧪 Input Wine Chemical Properties")

inputs = {}
for feature in features:
    inputs[feature] = st.number_input(
        feature.replace("_", " ").title(),
        value=float(df[feature].mean())
    )

input_df = pd.DataFrame([inputs])

# ---------------- Prediction ----------------
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = np.max(probabilities)

    st.success(f"🍷 Predicted Quality: **{prediction}**")
    st.write(f"🔢 Prediction Confidence: **{confidence:.2f}**")

# ---------------- Confusion Matrix ----------------
st.subheader("🧩 Confusion Matrix")

y_pred_test = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test, labels=["Low", "Medium", "High"])

fig2, ax2 = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "Medium", "High"]
)
disp.plot(ax=ax2)
st.pyplot(fig2)

