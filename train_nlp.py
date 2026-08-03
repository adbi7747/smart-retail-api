import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# 1. GENERATE CUSTOM INTENTS (Module B3 Requirement)
# ---------------------------------------------------------
intents_data = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey there", "Is anyone there?"], "responses": ["Hello! Welcome to our Smart Retail store. How can I help?"]},
        {"tag": "hours", "patterns": ["What are your hours?", "When do you open?", "Are you open today?"], "responses": ["We are open Monday to Saturday, 9 AM to 9 PM."]},
        {"tag": "returns", "patterns": ["Can I return an item?", "What is your return policy?", "I want a refund"], "responses": ["You can return any unworn item within 30 days with the receipt."]},
        {"tag": "shipping", "patterns": ["Do you deliver?", "How long does shipping take?", "Track my order"], "responses": ["Standard shipping takes 3-5 business days."]}
    ]
}

with open("intents.json", "w") as f:
    json.dump(intents_data, f)
print("✅ intents.json created!")

# ---------------------------------------------------------
# 2. TRAIN CHATBOT INTENT CLASSIFIER (Module B3 Requirement)
# ---------------------------------------------------------
X_chat = []
y_chat = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        X_chat.append(pattern)
        y_chat.append(intent['tag'])

chatbot_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
chatbot_pipeline.fit(X_chat, y_chat)

joblib.dump(chatbot_pipeline, "chatbot_model.pkl")
print("✅ Chatbot ML model trained and saved as chatbot_model.pkl!")

# ---------------------------------------------------------
# 3. TRAIN SENTIMENT ANALYSIS MODEL (Module B2 Requirement)
# ---------------------------------------------------------
reviews = [
    ("I absolutely love this product, it works great!", "Positive"),
    ("Terrible quality, it broke on the first day.", "Negative"),
    ("It is okay, nothing special but it does the job.", "Neutral"),
    ("Best purchase I have ever made.", "Positive"),
    ("Awful experience, I want a refund.", "Negative"),
    ("Shipping was fine, item is as described.", "Neutral")
]
df_sentiment = pd.DataFrame(reviews, columns=['review', 'sentiment'])

sentiment_pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
sentiment_pipeline.fit(df_sentiment['review'], df_sentiment['sentiment'])

joblib.dump(sentiment_pipeline, "sentiment_model.pkl")
print("✅ Sentiment ML model trained and saved as sentiment_model.pkl!")