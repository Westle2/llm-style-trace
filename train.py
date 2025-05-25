import os
import json
import random
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

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
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
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
    # 设置参数
    model_name = 'hfl/chinese-bert-wwm-ext'
    dataset_path = 'merge.jsonl'
    num_labels = 6 # qwen，gpt，human，deepseek，llama，文心一言
    '''
    这里做了六分类，或者先分出人类作者与大模型，再进行五分类。
    '''

    # 加载模型与分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 数据划分
    lines = load_dataset(dataset_path)
    random.shuffle(lines)
    split = int(0.9 * len(lines)) # 原模型是0.8, 内存不够了跑不动了
    train_dataset = prepare_dataset(lines[:split], tokenizer)
    eval_dataset = prepare_dataset(lines[split:], tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./bert_output",
        eval_strategy="epoch", # 有的transformer是evaluation_strategy字段
        save_strategy="epoch",
        learning_rate=3e-5,  # 原模型是2e-5,内存不够了
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
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
    # 保存模型
    trainer.save_model("saved_bert_model")

if __name__ == "__main__":
    main()
