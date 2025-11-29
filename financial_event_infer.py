

import argparse
import re
import os
import json
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Sequence, Features, Value
import evaluate
from dateutil import parser as dateparser

# ------------------------
# Utility helpers
# ------------------------

LABEL_LIST_NER = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-MONEY",
    "I-MONEY",
    "B-DATE",
    "I-DATE",
    "B-EVENT",
    "I-EVENT",
]

EVENT_LABELS = ["NONE", "MERGER", "ACQUISITION", "EARNINGS", "PARTNERSHIP"]

MONEY_REGEX = re.compile(r"\$?\s?\b(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?:\s?(?:million|billion|bn|m))?\b", flags=re.IGNORECASE)
DATE_SIMPLE_REGEX = re.compile(r"\b(?:Jan(?:uary)?|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\w\s,.-]*\d{2,4}\b")


def parse_money(text: str) -> List[str]:
    return [m.group(0) for m in MONEY_REGEX.finditer(text)]


def parse_dates(text: str) -> List[str]:
    res = []
    for m in DATE_SIMPLE_REGEX.finditer(text):
        res.append(m.group(0))
    # fallback: try dateutil parser on tokens
    words = text.split('\n')
    for seg in words:
        try:
            dt = dateparser.parse(seg, fuzzy=True)
            if dt:
                res.append(dt.date().isoformat())
        except Exception:
            pass
    return list(dict.fromkeys(res))


# ------------------------
# NER preprocessing helpers
# ------------------------

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subwords
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ------------------------
# Training / evaluation
# ------------------------

def train_ner(model_name_or_path, dataset, output_dir, epochs=3, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # dataset must have tokens and ner_tags
    tokenized = dataset.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path, num_labels=len(LABEL_LIST_NER)
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        preds = predictions.argmax(-1)
        true_preds = [[] for _ in range(len(labels))]
        true_labels = [[] for _ in range(len(labels))]
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] != -100:
                    true_labels[i].append(LABEL_LIST_NER[labels[i][j]])
                    true_preds[i].append(LABEL_LIST_NER[preds[i][j]])
        results = metric.compute(predictions=true_preds, references=true_labels)
        # return overall metrics
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    args = TrainingArguments(
   
    output_dir=output_dir,
    eval_strategy="epoch",   # <-- FIX
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2,

)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_cls(model_name_or_path, dataset, output_dir, epochs=3, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=len(EVENT_LABELS)
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": metric.compute(predictions=preds, references=p.label_ids)["accuracy"]}

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",

        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# ------------------------
# Inference pipeline
# ------------------------

def load_models_for_inference(ner_model_dir, cls_model_dir):
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir, use_fast=True)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_dir)

    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_dir, use_fast=True)
    cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_dir)

    ner_model.eval()
    cls_model.eval()

    return ner_tokenizer, ner_model, cls_tokenizer, cls_model


def ner_predict(text: str, tokenizer, model):
    # naive sentence -> tokens mapping
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids()

    entities = []
    cur_ent = None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        label_id = preds[idx]
        label = LABEL_LIST_NER[label_id]
        word = inputs.tokens()[idx]
        if label.startswith("B-"):
            if cur_ent:
                entities.append(cur_ent)
            cur_ent = {"type": label[2:], "tokens": [word]}
        elif label.startswith("I-") and cur_ent is not None:
            cur_ent["tokens"].append(word)
        else:
            if cur_ent:
                entities.append(cur_ent)
                cur_ent = None
    if cur_ent:
        entities.append(cur_ent)
    # postprocess tokens -> string
    for e in entities:
        e["text"] = tokenizer.convert_tokens_to_string(e["tokens"][1:]) if e["tokens"] and e["tokens"][0].startswith("##") else tokenizer.convert_tokens_to_string(e["tokens"])
        e.pop("tokens", None)
    return entities


def cls_predict(text: str, tokenizer, model):
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).item()
    return EVENT_LABELS[preds], torch.softmax(outputs.logits, dim=-1).squeeze()[preds].item()


def extract_structured_events(text: str, ner_tokenizer, ner_model, cls_tokenizer, cls_model):
    # run classifiers
    event_label, event_conf = cls_predict(text, cls_tokenizer, cls_model)
    entities = ner_predict(text, ner_tokenizer, ner_model)
    money = parse_money(text)
    dates = parse_dates(text)

    return {
        "text": text,
        "event": {"label": event_label, "score": float(event_conf)},
        "entities": entities,
        "money_mentions": money,
        "date_mentions": dates,
    }


# ------------------------
# Minimal dataset creation helper for toy demo
# ------------------------

def make_toy_datasets():
    # Create tiny toy datasets for demo/training
    ner_train = {
        "tokens": [["ACME", "Corp", "to", "acquire", "Globex", "Inc", "for", "$", "1", "billion", "."]],
        "ner_tags": [[1, 2, 0, 0, 1, 2, 0, 4, 4, 4, 0]],
    }
    ner_valid = ner_train
    ner_ds = DatasetDict({
        "train": Dataset.from_dict(ner_train),
        "validation": Dataset.from_dict(ner_valid),
    })

    cls_train = {
        "text": ["ACME Corp to acquire Globex Inc for $1 billion."],
        "label": [2],
    }
    cls_valid = cls_train
    cls_ds = DatasetDict({
        "train": Dataset.from_dict(cls_train),
        "validation": Dataset.from_dict(cls_valid),
    })

    return ner_ds, cls_ds


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ner", "cls", "infer", "toy"], required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--output_dir", default="./out")
    parser.add_argument("--ner_model", default=None)
    parser.add_argument("--cls_model", default=None)
    parser.add_argument("--text_file", default=None)
    args = parser.parse_args()

    if args.task == "toy":
        ner_ds, cls_ds = make_toy_datasets()
        print("Made toy datasets. Call with --task ner or cls to train on them.")
        return

    if args.task == "ner":
        # user should provide a HF dataset or we use toy
        print("Training NER (expects dataset with 'tokens' and 'ner_tags' lists). Using toy dataset in this demo.")
        ner_ds, _ = make_toy_datasets()
        train_ner(args.model_name_or_path, ner_ds, args.output_dir)
        return

    if args.task == "cls":
        print("Training classifier (expects dataset with 'text' and 'label'). Using toy dataset in this demo.")
        _, cls_ds = make_toy_datasets()
        train_cls(args.model_name_or_path, cls_ds, args.output_dir)
        return

    if args.task == "infer":
        if not args.ner_model or not args.cls_model:
            print("Provide --ner_model and --cls_model directories saved from training.")
            return
        ner_tokenizer, ner_model, cls_tokenizer, cls_model = load_models_for_inference(args.ner_model, args.cls_model)
        if args.text_file:
            with open(args.text_file, "r") as f:
                text = f.read()
            out = extract_structured_events(text, ner_tokenizer, ner_model, cls_tokenizer, cls_model)
            print(json.dumps(out, indent=2))
        else:
            print("Enter text (single line):")
            text = input().strip()
            out = extract_structured_events(text, ner_tokenizer, ner_model, cls_tokenizer, cls_model)
            print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
