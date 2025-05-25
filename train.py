import os
import json
import random
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from sklearn.preprocessing import LabelEncoder
import pickle

# 加载数据
def load_dataset(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    random.shuffle(lines)
    return lines

# 转换为HF Dataset
def prepare_dataset(lines, tokenizer):
    texts = [l['text'] for l in lines]
    labels = [l['label'] for l in lines]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    encodings['labels'] = labels
    return Dataset.from_dict(encodings)

# 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 主程序
def main():
    model_name = 'hfl/chinese-bert-wwm-ext'
    dataset_path = 'merge.jsonl'

    # 加载数据
    lines = load_dataset(dataset_path)

    # 将 "model" 字段编码为数字标签
    label_encoder = LabelEncoder()
    for line in lines:
        line['label'] = line['model']  # 直接把 model 放到 label 字段里（临时重命名）
    all_model_names = [line['label'] for line in lines]
    encoded_labels = label_encoder.fit_transform(all_model_names)
    for i, line in enumerate(lines):
        line['label'] = int(encoded_labels[i])

    # 保存 label_encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    num_labels = len(label_encoder.classes_)  # 自动获取类别数，例如6类

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 数据划分
    split = int(0.85 * len(lines))
    train_dataset = prepare_dataset(lines[:split], tokenizer)
    eval_dataset = prepare_dataset(lines[split:], tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./bert_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()
    trainer.save_model("saved_bert_model")

if __name__ == "__main__":
    main()