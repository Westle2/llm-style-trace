import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 模型路径
model_path = "saved_bert_model"

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# 加载 label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 推理函数
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_id = probs.argmax().item()
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    return pred_label, probs[0][pred_id].item()

# 示例
if __name__ == "__main__":
    example_text = "你干嘛哈哈哎哟"
    label, confidence = predict(example_text)
    print(f"预测标签: {label}, 置信度: {confidence:.4f}")
