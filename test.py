from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

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

# Ví dụ sử dụng hàm dự đoán
if __name__ == "__main__":
    new_comment = "đăng ký ngay"
    print(f"Dự đoán cho comment: {new_comment}")
    print(f"Kết quả: {predict_comment(new_comment)}")
