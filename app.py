from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = Flask(__name__, static_folder="static")
CORS(app)

# -----------------------------
# Global (lazy loaded)
# -----------------------------
tokenizer = None
model = None
id2label = {
    0: "Infrastructure_Road_Condition",
    1: "Waste_Management_Concern",
    2: "Land_Allotment_Query",
    3: "Infrastructure_Water_Supply_Issue",
    4: "Infrastructure_Power_Outage"
}

MODEL_NAME = "google/muril-base-cased"

# -----------------------------
# Load model lazily (IMPORTANT)
# -----------------------------
def load_model():
    global tokenizer, model
    if model is None:
        print("⬇️ Loading MuRIL model from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=5
        )
        model.eval()
        print("✅ Model loaded successfully")

# -----------------------------
# Utilities
# -----------------------------
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

def predict_intent(text):
    load_model()

    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "predicted_intent": id2label[pred.item()],
        "language_detected": detect_language(text),
        "confidence_score": round(confidence.item(), 4)
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/ping")
def ping():
    return jsonify({"status": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "predicted_intent": "unknown",
                "language_detected": "unknown",
                "confidence_score": 0.0
            })

        result = predict_intent(text)
        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "predicted_intent": "error",
            "language_detected": "error",
            "confidence_score": 0.0
        }), 500

@app.route("/")
def serve_html():
    return send_from_directory("static", "index.html")