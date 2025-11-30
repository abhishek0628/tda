
import argparse
import re
import json
import sys
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from dateutil import parser as dateparser

# ------------------------
# Constants
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

COMPANY_DB = [
    "TechNova Inc.",
    "GreenEnergy Ltd.",
    "GreenEnergy Ltd",
    "Apple",
    "Microsoft"
]

MONEY_REGEX = re.compile(r"\$?\s?\b(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?:\s?(?:million|billion|bn|m))?\b", flags=re.IGNORECASE)
DATE_REGEX = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4}\b")

# ------------------------
# Load models
# ------------------------
def load_models_for_inference(ner_model_dir, cls_model_dir):
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_dir, use_fast=True)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_dir)
    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_dir, use_fast=True)
    cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_dir)
    ner_model.eval()
    cls_model.eval()
    return ner_tokenizer, ner_model, cls_tokenizer, cls_model

# ------------------------
# NER prediction
# ------------------------
def ner_predict(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).squeeze().tolist()
    word_ids = inputs.word_ids(batch_index=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    cur_ent = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        label_id = preds[idx]
        label = LABEL_LIST_NER[label_id]
        token = tokens[idx]
        token_clean = token[2:] if token.startswith("##") else token

        if label.startswith("B-"):
            if cur_ent:
                entities.append(cur_ent)
            cur_ent = {"type": label[2:], "text": token_clean}
        elif label.startswith("I-") and cur_ent is not None:
            cur_ent["text"] += " " + token_clean
        else:
            if cur_ent:
                entities.append(cur_ent)
                cur_ent = None

    if cur_ent:
        entities.append(cur_ent)
    return entities

# ------------------------
# Filter companies from DB
# ------------------------
def filter_companies(entities):
    companies = []
    for e in entities:
        if e['type'] == 'COMPANY':
            ent_text_clean = e['text'].replace('.', '').lower()
            for comp in COMPANY_DB:
                comp_clean = comp.replace('.', '').lower()
                if comp_clean in ent_text_clean or ent_text_clean in comp_clean:
                    companies.append(comp)
                    break
    return list(dict.fromkeys(companies))

# ------------------------
# Extract money
# ------------------------
def parse_money(text: str) -> List[str]:
    matches = MONEY_REGEX.findall(text)
    clean = []
    for m in matches:
        m = re.sub(r"(?i)(\d)(million|billion|m|bn)", r"\1 \2", m)
        clean.append(m.strip())
    return clean

# ------------------------
# Extract date
# ------------------------
def parse_date(text: str) -> str:
    # try regex first
    m = DATE_REGEX.search(text)
    if m:
        return m.group(0)
    # fallback to dateparser
    try:
        dt = dateparser.parse(text, fuzzy=True)
        if dt:
            return dt.strftime("%d %B %Y")
    except Exception:
        pass
    return "None"

# ------------------------
# Extract deal description
# ------------------------
def extract_deal(text: str) -> str:
    # Look for keywords: partnership, merger, acquisition
    match = re.search(r"(?:partnership|merger|acquisition).*?on.*?to\s+(.*?)(?:\.|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "None"

# ------------------------
# Classification prediction
# ------------------------
def cls_predict(text: str, tokenizer, model):
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).item()
    score = torch.softmax(outputs.logits, dim=-1).squeeze()[preds].item()
    return EVENT_LABELS[preds], float(score)

# ------------------------
# Extract structured events
# ------------------------
def extract_structured_events(text: str, ner_tokenizer, ner_model, cls_tokenizer, cls_model):
    event_label, event_score = cls_predict(text, cls_tokenizer, cls_model)
    entities = ner_predict(text, ner_tokenizer, ner_model)
    companies = filter_companies(entities)
    deal = extract_deal(text)
    money = parse_money(text)
    date = parse_date(text)
    return {
        "event": {"label": event_label, "score": event_score},
        "companies": companies if companies else "None",
        "deal": deal,
        "money_investment": money[0] if money else "None",
        "date": date
    }

# ------------------------
# Format key-value
# ------------------------
def format_as_key_value(result):
    lines = []
    lines.append(f"event_type: {result['event']['label']}")
    lines.append(f"event_score: {result['event']['score']:.2f}")
    lines.append(f"companies: {', '.join(result['companies']) if isinstance(result['companies'], list) else result['companies']}")
    lines.append(f"deal: {result['deal']}")
    lines.append(f"money_investment: {result['money_investment']}")
    lines.append(f"date: {result['date']}")
    return "\n".join(lines)

# ------------------------
# Main CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["infer"], required=True)
    parser.add_argument("--ner_model", required=True)
    parser.add_argument("--cls_model", required=True)
    parser.add_argument("--text_file", required=True)
    args = parser.parse_args()

    ner_tokenizer, ner_model, cls_tokenizer, cls_model = load_models_for_inference(
        args.ner_model, args.cls_model
    )

    with open(args.text_file, 'r') as f:
        text = f.read()

    result = extract_structured_events(text, ner_tokenizer, ner_model, cls_tokenizer, cls_model)
    print(format_as_key_value(result))
