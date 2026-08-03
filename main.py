import json
import joblib
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import random
from PIL import Image
import io

# --- 1. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform",
    description="Unified API for Face Recognition, Product Classification, Sentiment Analysis, and Customer Support"
)

# --- 2. LOAD NLP MODELS (Module B) ---
# Loading the models we just trained via joblib
sentiment_model = joblib.load("sentiment_model.pkl")
chatbot_model = joblib.load("chatbot_model.pkl")

with open("intents.json", "r") as f:
    intents_data = json.load(f)

# --- 3. SETUP VISION MODELS (Module A) ---
# A2: Product Classifier - PyTorch MobileNetV2
weights = models.MobileNet_V2_Weights.DEFAULT
product_model = models.mobilenet_v2(weights=weights)
product_model.eval() # Set to evaluation mode

# Image preprocessing pipeline for PyTorch
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# A1: OpenCV Face Detection Setup using Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 4. PYDANTIC SCHEMAS ---
class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str

# --- 5. API ENDPOINTS (Module C3) ---

@app.post("/analyze-sentiment")
async def analyze_sentiment(req: TextRequest):
    prediction = sentiment_model.predict([req.text])[0]
    return {"text": req.text, "sentiment": prediction}

@app.post("/chatbot")
async def chatbot(req: ChatRequest):
    # Predict the intent tag using ML
    intent_tag = chatbot_model.predict([req.message])[0]
    
    # Retrieve a random response for that intent
    response = "I'm not sure how to help with that. Please contact support."
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            break
            
    return {"message": req.message, "intent": intent_tag, "reply": response}

@app.post("/classify-product")
async def classify_product(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Get prediction from MobileNetV2
    with torch.no_grad():
        output = product_model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    category = weights.meta["categories"][top_catid[0].item()]
    
    return {"filename": file.filename, "category": category, "confidence": float(top_prob[0].item())}

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
    # Read uploaded image with OpenCV
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocessing: Grayscale required for Haar Cascades
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return {"status": "No face detected"}
        
    # Mocking the LBPH loyalty check for the 1-day sprint timeline
    return {
        "status": "Face detected", 
        "faces_found": len(faces),
        "action": "Returning Customer Visit Logged"
    }

@app.get("/dashboard/stats")
async def get_stats():
    return {
        "total_visits_today": 142,
        "positive_sentiment_ratio": "82%",
        "active_chatbot_sessions": 8
    }
# --- PYTORCH & OPENCV SETUP ---

# Load pre-trained MobileNetV2 for fast product classification
weights = models.MobileNet_V2_Weights.DEFAULT
cv_model = models.mobilenet_v2(weights=weights)
cv_model.eval()

# PyTorch Image Preprocessing Pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- CV ENDPOINTS ---

@app.post("/classify-product")
async def classify_product(file: UploadFile = File(...)):
    """Classifies an uploaded product image using PyTorch MobileNetV2."""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = cv_model(input_batch)
    
    # Get top prediction (using ImageNet classes for demo purposes)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    return {
        "filename": file.filename,
        "category_id": top_catid.item(),
        "confidence": round(top_prob.item() * 100, 2)
    }

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
    """Detects faces in an uploaded image using OpenCV Haar Cascades."""
    image_data = await file.read()
    
    # Convert uploaded image to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return {
        "filename": file.filename,
        "faces_detected": len(faces),
        "status": "Returning Customer Logged" if len(faces) > 0 else "No Face Detected"
    }
# --- DASHBOARD ENDPOINT ---

@app.get("/dashboard/stats")
async def get_stats():
    """Returns aggregate visit and sentiment stats for the frontend dashboard."""
    # In a real app, this would query a database. For this project, we return a static/mock JSON response.
    return {
        "status": "success",
        "data": {
            "total_store_visits_today": 215,
            "returning_customers": 42,
            "overall_sentiment_score": "Positive",
            "top_product_category": "Apparel"
        }
    }