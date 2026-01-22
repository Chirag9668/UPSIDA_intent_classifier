from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

# ----------------------------
# Load YOUR trained MuRIL model
# ----------------------------
MODEL_PATH = "model"   # <-- local folder, NOT GitHub

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "❌ model/ folder not found. "
        "Please unzip your trained MuRIL model inside the project."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Label mapping (same as training)
id2label = {
    0: "Infrastructure_Road_Condition",
    1: "Waste_Management_Concern",
    2: "Land_Allotment_Query",
    3: "Infrastructure_Water_Supply_Issue",
    4: "Infrastructure_Power_Outage"
}

# ----------------------------
# Text utilities
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_language(text):
    devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
    latin = sum('a' <= ch.lower() <= 'z' for ch in text)

    if devanagari and latin:
        return "Hinglish"
    elif devanagari:
        return "Hindi"
    elif latin:
        return "English"
    return "Unknown"

# ----------------------------
# Inference
# ----------------------------
def predict_intent(text):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "text_input": text,
        "predicted_intent": id2label[pred.item()],
        "confidence_score": round(confidence.item(), 4),
        "language_detected": detect_language(text)
    }

# ----------------------------
# Routes
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    invalid_inputs = [
        "hi", "hello", "how are you", "who are you",
        "what is your name", "thanks", "thank you",
        "ok", "bye", "help", "?", "!"
    ]

    if not text or text.lower() in invalid_inputs or len(text.split()) < 3:
        return make_response(
            "Sorry, I can't understand your problem. Please type your complaint.",
            200
        )

    result = predict_intent(text)
    return jsonify(result)

@app.route("/")
def serve_html():
    return send_from_directory("static", "index.html")

# ----------------------------
# LOCAL RUN ONLY
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)