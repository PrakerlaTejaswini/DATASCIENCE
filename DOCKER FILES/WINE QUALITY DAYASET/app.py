from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Wine Quality Prediction</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f5f5f5;
        }
        .container {
            width: 420px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
        }
        input {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #7b2cbf;
            color: white;
            border: none;
            font-size: 16px;
        }
        h2, h3 {
            text-align: center;
        }
    </style>
</head>

<body>
    <div class="container">
        <h2>🍷 Wine Quality Prediction</h2>
        <form method="post">
            <input name="fixed_acidity" placeholder="Fixed Acidity" required>
            <input name="volatile_acidity" placeholder="Volatile Acidity" required>
            <input name="citric_acid" placeholder="Citric Acid" required>
            <input name="residual_sugar" placeholder="Residual Sugar" required>
            <input name="chlorides" placeholder="Chlorides" required>
            <input name="free_sulfur_dioxide" placeholder="Free Sulfur Dioxide" required>
            <input name="total_sulfur_dioxide" placeholder="Total Sulfur Dioxide" required>
            <input name="density" placeholder="Density" required>
            <input name="pH" placeholder="pH" required>
            <input name="sulphates" placeholder="Sulphates" required>
            <input name="alcohol" placeholder="Alcohol" required>
            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <h3>Result: {{ prediction }}</h3>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        features = np.array([[
            float(request.form["fixed_acidity"]),
            float(request.form["volatile_acidity"]),
            float(request.form["citric_acid"]),
            float(request.form["residual_sugar"]),
            float(request.form["chlorides"]),
            float(request.form["free_sulfur_dioxide"]),
            float(request.form["total_sulfur_dioxide"]),
            float(request.form["density"]),
            float(request.form["pH"]),
            float(request.form["sulphates"]),
            float(request.form["alcohol"])
        ]])

        result = model.predict(features)[0]
        prediction = "Good Wine 🍷" if result == 1 else "Bad Wine ❌"

    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
