from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load vectorizer and model
with open("spam_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>SMS Spam Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 50px; }
            h2 { color: #333; }
            textarea { width: 60%; padding: 10px; font-size: 14px; }
            input[type=submit] { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            input[type=submit]:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }
        </style>
    </head>
    <body>
        <h2>📩 SMS Spam Detection</h2>
        <form method="POST" action="/predict">
            <textarea name="message" rows="4" cols="50" placeholder="Enter your SMS here"></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        message = data.get('message', '')
    else:
        message = request.form.get('message', '')

    if not message.strip():
        return jsonify({"error": "Message is empty"}), 400

    message_vect = vectorizer.transform([message])
    prediction = model.predict(message_vect)[0]

    # If form submission, return styled HTML
    if not request.is_json:
        return f'''
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 50px; }}
                .result {{ margin-top: 20px; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }}
                h3 {{ color: #333; }}
                p {{ font-size: 16px; }}
            </style>
        </head>
        <body>
            <div class="result">
                <h3>Message:</h3>
                <p>{message}</p>
                <h3>Prediction:</h3>
                <p style="color:{'red' if prediction=='spam' else 'green'}; font-weight:bold;">{prediction}</p>
                <a href="/">🔙 Back</a>
            </div>
        </body>
        </html>
        '''

    # If JSON request, return JSON
    return jsonify({"message": message, "prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)