# src/train_common.py

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch


MODEL_NAME = "distilbert-base-uncased"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def load_model(num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def create_trainer(model, tokenized_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=20,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
