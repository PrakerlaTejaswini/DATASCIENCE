import pandas as pd

# Load dataset with latin-1 encoding
df = pd.read_csv("spam.csv", usecols=["v1","v2"], encoding="latin-1")
df = df.rename(columns={"v1":"label","v2":"text"})

# Encode labels
df["label"] = df["label"].map({"ham":0,"spam":1})

# Remove empty messages & lowercase
df = df[df["text"].notnull()]
df["text"] = df["text"].str.lower()

# Save cleaned CSV
df.to_csv("sms_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as sms_cleaned.csv")
