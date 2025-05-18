from flask import Flask, request, jsonify
from flask_cors import CORS  # import thêm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

app = Flask(__name__)
CORS(app)  # bật CORS cho toàn app

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("./results")
model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("./results")

def predict_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    label = "Spam" if pred == 1 else "Non-Spam"
    return label, confidence

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comment = data.get("comment", "")
    if not comment:
        return jsonify({"error": "Missing 'comment' field"}), 400

    label, confidence = predict_comment(comment)
    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
