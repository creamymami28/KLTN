import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

# Đọc dữ liệu của bạn
df = pd.read_csv("spam_detection_dataset_vi.csv")  # Thay đổi đường dẫn nếu cần
df["label"] = df["label"].astype(int)

# Tách dữ liệu train và validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["comment"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Tải tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tạo Dataset cho PyTorch
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

# Tải mô hình và huấn luyện
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./results",  # Thư mục để lưu mô hình
    evaluation_strategy="epoch",  # Đánh giá mô hình sau mỗi epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",  # Đảm bảo mô hình sẽ được lưu sau mỗi epoch
    logging_dir='./logs',  # Đường dẫn cho các file log
    save_total_limit=1,  # Giới hạn số lượng checkpoint lưu lại
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Lưu mô hình đã huấn luyện
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

# print("Thư mục kết quả:", os.listdir('./results'))