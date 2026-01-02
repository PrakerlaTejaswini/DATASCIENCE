import streamlit as st
import pickle
import numpy as np

# Load trained models
with open(r"C:\Users\LENOVO\Documents\AIML PROJECTS\iris_model.pkl", "rb") as file:
    models = pickle.load(file)

st.title("🌸 Iris Flower Classification")
st.write("Predict Iris species using ML models")

# Model selection
model_name = st.selectbox(
    "Choose Model",
    list(models.keys())
)

# Input fields
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if st.button("Predict"):
    model = models[model_name]
    prediction = model.predict(input_data)[0]

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"🌼 Predicted Species: **{species[prediction]}**")
