from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import json


app = Flask(__name__)

# Chọn device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình đã huấn luyện
model = DistilBertForSequenceClassification.from_pretrained("./results")
model.to(device)

# Tải tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Hàm dự đoán
def predict_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Spam" if prediction == 1 else "Non-Spam"

# Endpoint nhận request dự đoán từ extension
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # In ra thông tin yêu cầu để kiểm tra
        print("Headers:", request.headers)
        print("Data received:", request.data)

        # Giải mã chuỗi byte thành chuỗi UTF-8 và phân tích dữ liệu JSON
        data = json.loads(request.data.decode('utf-8'))

        # Kiểm tra xem dữ liệu có hợp lệ hay không
        if not data or "comment" not in data:
            return jsonify({"error": "Invalid or missing 'comment' in request"}), 400

        # Lấy giá trị bình luận từ dữ liệu JSON
        comment = data["comment"]

        # Giả sử mô hình dự đoán là "Spam"
        prediction = "Spam"  # Đây chỉ là ví dụ, thay thế với mô hình thật

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
