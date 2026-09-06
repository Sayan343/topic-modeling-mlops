"""
app.py
A small Flask API that loads the trained LDA model and serves
topic predictions for new text.

Endpoints:
  GET  /health            -> simple check that the service is alive
  POST /predict            -> {"text": "..."} -> topic prediction
"""

import pickle
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("lda_model.pkl", "rb") as f:
    lda = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("topic_labels.json") as f:
    topic_labels = json.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "Send JSON like {'text': 'your text here'}"}), 400

    text = payload["text"]
    if not text.strip():
        return jsonify({"error": "text field is empty"}), 400

    vec = vectorizer.transform([text])
    topic_distribution = lda.transform(vec)[0]

    best_topic_idx = int(topic_distribution.argmax())
    confidence = float(topic_distribution[best_topic_idx])

    return jsonify({
        "topic_id": best_topic_idx,
        "topic_label": topic_labels[str(best_topic_idx)]["label"],
        "keywords": topic_labels[str(best_topic_idx)]["keywords"],
        "confidence": round(confidence, 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
