import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load cleaned dataset
df = pd.read_csv(r"sms_cleaned.csv")
X = df["text"]
y = df["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model + vectorizer in same folder
with open("spam_model.pkl","wb") as f:
    pickle.dump((vectorizer, model), f)
print("✅ Model trained and saved as spam_model.pkl")
