from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = Flask(__name__, static_folder="static")
CORS(app)

# Load MuRIL model from local directory
tokenizer = AutoTokenizer.from_pretrained("muril_model")
model = AutoModelForSequenceClassification.from_pretrained("muril_model")
model.eval()
id2label = model.config.id2label

# Preprocess input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Simple language detection
def detect_language(text):
    devanagari = sum('\u0900' <= ch <= '\u097F' for ch in text)
    latin = sum('a' <= ch.lower() <= 'z' for ch in text)
    if devanagari > 0 and latin > 0:
        return "Hinglish"
    elif devanagari > 0:
        return "Hindi"
    elif latin > 0:
        return "English"
    return "Unknown"

# Run the model prediction
def predict_intent(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return {
        "text_input": text,
        "language_detected": detect_language(text),
        "predicted_intent": id2label[pred.item()],
        "confidence_score": round(conf.item(), 2)
    }

# Main prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip().lower()

    # Handle invalid or unrelated inputs
    invalid_inputs = [
        "hi", "hello", "how are you", "who are you", "what is your name", "thanks",
        "thank you", "good morning", "good night", "ok", "bye", "help", "?", "!"
    ]

    if text in invalid_inputs or len(text.split()) < 3:
        return make_response(
            "Sorry, I can't understand your problem. Please type your complaint. Thank you.",
            200
        )

    # Valid complaint input, run prediction
    result = predict_intent(data["text"])
    return jsonify(result)

# Serve the frontend HTML
@app.route("/")
def serve_html():
    return send_from_directory("static", "index.html")

if __name__ == '__main__':
    app.run(debug=True)
