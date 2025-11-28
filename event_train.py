### FILE: event_train.py
"""
Train a simple span-pair classifier that takes a trigger span and candidate argument spans and predicts role labels.
This is a simplified baseline: construct input as: [CLS] sentence [SEP] <trigger-span> [SEP] <arg-span>
and fine-tune a sequence classifier on that input.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from utils import load_jsonl, normalize_date


def build_pairs(jsonl_path, tokenizer, max_len=256):
    examples = {'input_text': [], 'label': []}
    label_set = set()
    for r in load_jsonl(jsonl_path):
        text = r['text']
        for ev in r.get('events', []):
            trigger = ev['trigger']
            for arg in ev.get('arguments', []):
                combined = f"[TRG] {trigger['text']} [ARG] {arg['text']} [SENT] {text}"
                role = arg['role']
                examples['input_text'].append(combined)
                examples['label'].append(role)
                label_set.add(role)
    labels = sorted(label_set)
    label2id = {l:i for i,l in enumerate(labels)}
    # map labels to ids
    examples['labels'] = [label2id[l] for l in examples['label']]
    ds = Dataset.from_dict({'text': examples['input_text'], 'label': examples['labels']})
    return ds, labels


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds, labels = build_pairs(args.train_file, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels))

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized = ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        evaluation_strategy='no',
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default='./event_model')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train(args)
