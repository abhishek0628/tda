# # Project: Transformer-Based Financial Event Extraction
# # Repository structure shown as multiple files below. Save each section into its own file named after the header.

# ### FILE: README.md
# """
# Transformer-Based Financial Event Extraction
# ===========================================

# This repo contains a two-stage, practical baseline for extracting financial events from news text:

# 1) Named Entity Recognition (NER) token-classifier to extract company names, dates, money, and other entities using a pretrained RoBERTa/BERT model.
# 2) Event classification & argument linking: pairwise span-based classifier that classifies whether a candidate trigger span + argument spans form a specific event (e.g. merger, acquisition, earnings, partnership).

# Outputs structured JSON records: {event_type, trigger_text, trigger_span, arguments: [{role, text, span}], confidence}

# Requirements
# ------------
# Python 3.8+
# Install with:

#     pip install -r requirements.txt

# Files
# -----
# - requirements.txt
# - dataset.py           # dataset loading + tokenization utilities
# - ner_train.py         # train/eval NER model (token classification)
# - event_train.py       # train/eval event classifier (span pair classifier)
# - inference.py         # end-to-end inference script producing JSON output
# - utils.py             # helper functions
# - sample_data.jsonl    # sample dataset format (JSONL)

# How to run
# ----------
# 1. Prepare data as JSONL (see sample_data.jsonl)
# 2. Train NER:
#     python ner_train.py --train_file sample_data.jsonl --model_name roberta-base --output_dir ./ner_model
# 3. Train event classifier (uses NER outputs or gold spans):
#     python event_train.py --train_file sample_events.jsonl --ner_model ./ner_model --output_dir ./event_model
# 4. Inference:
#     python inference.py --model_ner ./ner_model --model_event ./event_model --input_file test_news.jsonl --output_file results.jsonl

# Notes
# -----
# This is a baseline with clarity prioritized over cutting-edge performance. Real-world use should augment with domain-specific pretraining, distant supervision, data augmentation, and more sophisticated joint modeling.
# """

# ### FILE: requirements.txt
# """
# transformers>=4.30.0
# datasets>=2.0.0
# torch>=1.13.0
# tqdm
# seqeval
# spacy
# python-dateutil
# fuzzywuzzy
# python-Levenshtein
# """

# ### FILE: sample_data.jsonl
# """
# # Example records for NER (one sentence per line). Use BIO tags for token labels when creating token-level training.
# {
#   "id": "doc1_sent1",
#   "text": "Acme Corp agreed to acquire Beta Systems for $120 million on January 4, 2025.",
#   "entities": [
#     {"type": "ORG", "text": "Acme Corp", "start": 0, "end": 9},
#     {"type": "EVENT", "text": "acquire", "start": 22, "end": 29},
#     {"type": "ORG", "text": "Beta Systems", "start": 33, "end": 45},
#     {"type": "MONEY", "text": "$120 million", "start": 50, "end": 63},
#     {"type": "DATE", "text": "January 4, 2025", "start": 67, "end": 82}
#   ],
#   "events": [
#     {"type": "ACQUISITION", "trigger": {"text": "acquire", "start": 22, "end": 29}, "arguments": [{"role": "ACQUIRER", "text": "Acme Corp", "start": 0, "end": 9}, {"role": "TARGET", "text": "Beta Systems", "start": 33, "end": 45}, {"role": "PRICE", "text": "$120 million", "start": 50, "end": 63}, {"role": "DATE", "text": "January 4, 2025", "start": 67, "end": 82}] }
#   ]
# }
# """

# # ### FILE: utils.py
# # """
# # Helper utilities for tokenization, span conversion, and IO.
# # """
# # import json
# # from typing import List, Tuple, Dict

# # from dateutil import parser as dateparser


# # def load_jsonl(path: str):
# #     with open(path, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             if line.strip():
# #                 yield json.loads(line)


# # def save_jsonl(path: str, records):
# #     with open(path, 'w', encoding='utf-8') as f:
# #         for r in records:
# #             f.write(json.dumps(r, ensure_ascii=False) + "\n")


# # def normalize_date(text: str):
# #     try:
# #         dt = dateparser.parse(text, fuzzy=True)
# #         return dt.date().isoformat()
# #     except Exception:
# #         return text


# # def charspan_to_tokenspan(char_start: int, char_end: int, token_offsets: List[Tuple[int,int]]):
# #     # token_offsets is list of (start_char, end_char) for each token
# #     ts, te = None, None
# #     for i,(s,e) in enumerate(token_offsets):
# #         if ts is None and char_start >= s and char_start < e:
# #             ts = i
# #         if te is None and char_end > s and char_end <= e:
# #             te = i
# #     # fallback: find nearest
# #     if ts is None:
# #         for i,(s,e) in enumerate(token_offsets):
# #             if s >= char_start:
# #                 ts = i
# #                 break
# #     if te is None:
# #         for i,(s,e) in enumerate(reversed(token_offsets)):
# #             if e <= char_end:
# #                 te = len(token_offsets)-1 - i
# #                 break
# #     return ts, te


# ### FILE: dataset.py
# # """
# # Dataset utilities using Hugging Face datasets and tokenizers.
# # """
# # import os
# # from typing import List, Dict
# # from datasets import Dataset
# # from transformers import AutoTokenizer
# # from utils import load_jsonl, charspan_to_tokenspan


# # def load_ner_dataset(jsonl_path: str, model_name: str, label_list: List[str]=None):
# #     # Produces a HuggingFace Dataset with fields: tokens, labels (BIO)
# #     raw = list(load_jsonl(jsonl_path))
# #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# #     examples = {
# #         'tokens': [],
# #         'labels': []
# #     }

# #     # Build label mapping if not provided
# #     labels_set = set()
# #     for r in raw:
# #         for ent in r.get('entities', []):
# #             labels_set.add(ent['type'])
# #     labels = sorted(labels_set) if label_list is None else label_list
# #     label_to_id = {l:i for i,l in enumerate(labels)}

# #     for r in raw:
# #         txt = r['text']
# #         enc = tokenizer(txt, truncation=True, return_offsets_mapping=True)
# #         offsets = enc['offset_mapping']
# #         tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
# #         token_labels = ['O'] * len(tokens)

# #         for ent in r.get('entities', []):
# #             # map char span to token span
# #             start, end = ent['start'], ent['end']
# #             ts, te = charspan_to_tokenspan(start, end, offsets)
# #             if ts is None or te is None:
# #                 continue
# #             token_labels[ts] = 'B-' + ent['type']
# #             for idx in range(ts+1, te+1):
# #                 token_labels[idx] = 'I-' + ent['type']

# #         examples['tokens'].append(tokens)
# #         examples['labels'].append(token_labels)

# #     ds = Dataset.from_dict(examples)
# #     return ds, tokenizer, labels


# ### FILE: ner_train.py
# """
# Train a token classification model (NER) using Hugging Face Trainer.
# """
# import argparse
# from datasets import ClassLabel, load_metric
# from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
# from transformers import AutoTokenizer
# import numpy as np

# from dataset import load_ner_dataset


# def align_labels_with_tokens(labels, word_ids):
#     new_labels = []
#     for i, word_id in enumerate(word_ids):
#         if word_id is None:
#             new_labels.append(-100)
#         else:
#             label = labels[word_id]
#             # we assume labels given per token already here; if using word-level labels you'd convert
#             new_labels.append(label)
#     return new_labels


# def convert_bio_to_ids(label_list):
#     # label_list is set like ['B-ORG', 'I-ORG', 'O', ...]. We'll create mapping
#     unique = sorted(set(label_list))
#     id_map = {l:i for i,l in enumerate(unique)}
#     return unique, id_map


# def train(args):
#     ds, tokenizer, labels = load_ner_dataset(args.train_file, args.model_name)

#     # Flatten labels to unique set
#     flat = set()
#     for ll in ds['labels']:
#         for x in ll:
#             flat.add(x)
#     unique_labels = sorted(flat)
#     label2id = {l:i for i,l in enumerate(unique_labels)}
#     id2label = {i:l for l,i in label2id.items()}

#     model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

#     def tokenize_and_align(examples):
#         tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, return_tensors=None)
#         labels_aligned = []
#         for i, labels_seq in enumerate(examples['labels']):
#             word_ids = tokenized_inputs.word_ids(batch_index=i)
#             aligned = align_labels_with_tokens(labels_seq, word_ids)
#             # convert label strings to ids
#             aligned_ids = [label2id[l] if (l != -100 and l in label2id) else -100 if l== -100 else label2id.get(l, -100) for l in aligned]
#             labels_aligned.append(aligned_ids)
#         tokenized_inputs['labels'] = labels_aligned
#         return tokenized_inputs

#     tokenized = ds.map(tokenize_and_align, batched=True)

#     data_collator = DataCollatorForTokenClassification(tokenizer)
#     metric = load_metric('seqeval')

#     def compute_metrics(p):
#         predictions, labels = p
#         preds = np.argmax(predictions, axis=2)
#         true_labels = [[id2label[l] for l in lab if l != -100] for lab in labels]
#         true_preds = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
#         results = metric.compute(predictions=true_preds, references=true_labels)
#         return {
#             'precision': results['overall_precision'],
#             'recall': results['overall_recall'],
#             'f1': results['overall_f1']
#         }

#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=8,
#         num_train_epochs=args.epochs,
#         logging_steps=50,
#         save_strategy='epoch',
#         evaluation_strategy='epoch'
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized,
#         eval_dataset=tokenized,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics
#     )

#     trainer.train()
#     trainer.save_model(args.output_dir)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_file', type=str, required=True)
#     parser.add_argument('--model_name', type=str, default='roberta-base')
#     parser.add_argument('--output_dir', type=str, default='./ner_model')
#     parser.add_argument('--epochs', type=int, default=3)
#     args = parser.parse_args()
#     train(args)


# # ### FILE: event_train.py
# # """
# # Train a simple span-pair classifier that takes a trigger span and candidate argument spans and predicts role labels.
# # This is a simplified baseline: construct input as: [CLS] sentence [SEP] <trigger-span> [SEP] <arg-span>
# # and fine-tune a sequence classifier on that input.
# # """
# # import argparse
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# # from datasets import Dataset
# # import numpy as np
# # from utils import load_jsonl, normalize_date


# # def build_pairs(jsonl_path, tokenizer, max_len=256):
# #     examples = {'input_text': [], 'label': []}
# #     label_set = set()
# #     for r in load_jsonl(jsonl_path):
# #         text = r['text']
# #         for ev in r.get('events', []):
# #             trigger = ev['trigger']
# #             for arg in ev.get('arguments', []):
# #                 combined = f"[TRG] {trigger['text']} [ARG] {arg['text']} [SENT] {text}"
# #                 role = arg['role']
# #                 examples['input_text'].append(combined)
# #                 examples['label'].append(role)
# #                 label_set.add(role)
# #     labels = sorted(label_set)
# #     label2id = {l:i for i,l in enumerate(labels)}
# #     # map labels to ids
# #     examples['labels'] = [label2id[l] for l in examples['label']]
# #     ds = Dataset.from_dict({'text': examples['input_text'], 'label': examples['labels']})
# #     return ds, labels


# # def train(args):
# #     tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
# #     ds, labels = build_pairs(args.train_file, tokenizer)

# #     model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels))

# #     def tokenize_fn(examples):
# #         return tokenizer(examples['text'], truncation=True, padding='max_length')

# #     tokenized = ds.map(tokenize_fn, batched=True)

# #     training_args = TrainingArguments(
# #         output_dir=args.output_dir,
# #         per_device_train_batch_size=8,
# #         num_train_epochs=args.epochs,
# #         evaluation_strategy='no',
# #         save_strategy='epoch'
# #     )

# #     trainer = Trainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=tokenized
# #     )

# #     trainer.train()
# #     trainer.save_model(args.output_dir)


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--train_file', type=str, required=True)
# #     parser.add_argument('--model_name', type=str, default='roberta-base')
# #     parser.add_argument('--output_dir', type=str, default='./event_model')
# #     parser.add_argument('--epochs', type=int, default=3)
# #     args = parser.parse_args()
# #     train(args)


# # ### FILE: inference.py
# # """
# # End-to-end inference: run NER to get candidate spans, then run event classifier to assign roles and assemble events.
# # """
# # import argparse
# # from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
# # from utils import load_jsonl, save_jsonl, normalize_date


# # def extract_spans_ner(ner_model_dir, texts):
# #     ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_dir)
# #     tokenizer = AutoTokenizer.from_pretrained(ner_model_dir, use_fast=True)
# #     nlp = pipeline('ner', model=ner_model, tokenizer=tokenizer, aggregation_strategy='simple')
# #     all_spans = []
# #     for t in texts:
# #         preds = nlp(t)
# #         # preds include {'entity_group','score','word','start','end'}
# #         spans = [{'type': p['entity_group'], 'text': p['word'], 'start': p['start'], 'end': p['end'], 'score': p['score']} for p in preds]
# #         all_spans.append(spans)
# #     return all_spans


# # def run_event_classifier(event_model_dir, tokenizer, trigger_text, arg_text, sentence):
# #     model = AutoModelForSequenceClassification.from_pretrained(event_model_dir)
# #     # tokenizer passed in from event model
# #     combined = f"[TRG] {trigger_text} [ARG] {arg_text} [SENT] {sentence}"
# #     inputs = tokenizer(combined, truncation=True, return_tensors='pt')
# #     logits = model(**inputs).logits
# #     probs = logits.softmax(-1).detach().cpu().numpy()[0]
# #     label_id = probs.argmax()
# #     # label string mapping may not be saved; assume user saved labels separately in production.
# #     return label_id, float(probs[label_id])


# # def assemble_events(texts, ner_model_dir, event_model_dir, output_file):
# #     texts_list = [t for t in texts]
# #     spans_list = extract_spans_ner(ner_model_dir, texts_list)
# #     # load tokenizer used for event classifier
# #     tokenizer = AutoTokenizer.from_pretrained(event_model_dir, use_fast=True)
# #     model = AutoModelForSequenceClassification.from_pretrained(event_model_dir)

# #     results = []
# #     for text, spans in zip(texts_list, spans_list):
# #         # naive trigger selection: verbs or EVENT labels; otherwise use all spans as triggers
# #         triggers = [s for s in spans if s['type'] in ('EVENT', 'VERB')]
# #         if not triggers:
# #             triggers = spans

# #         events_for_text = []
# #         for trg in triggers:
# #             args_candidates = spans
# #             arguments = []
# #             for arg in args_candidates:
# #                 combined = f"[TRG] {trg['text']} [ARG] {arg['text']} [SENT] {text}"
# #                 inputs = tokenizer(combined, truncation=True, return_tensors='pt')
# #                 logits = model(**inputs).logits
# #                 probs = logits.softmax(-1).detach().cpu().numpy()[0]
# #                 label_id = int(probs.argmax())
# #                 score = float(probs[label_id])
# #                 if score > 0.5:  # threshold; in practice tune
# #                     arguments.append({'role_id': label_id, 'role_confidence': score, 'text': arg['text'], 'span': [arg['start'], arg['end'], arg['type']]})
# #             if arguments:
# #                 events_for_text.append({'trigger': trg['text'], 'trigger_span': [trg['start'], trg['end']], 'arguments': arguments})
# #         results.append({'text': text, 'events': events_for_text})

# #     save_jsonl(output_file, results)
# #     return results


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--model_ner', type=str, required=True)
# #     parser.add_argument('--model_event', type=str, required=True)
# #     parser.add_argument('--input_file', type=str, required=True)
# #     parser.add_argument('--output_file', type=str, default='results.jsonl')
# #     args = parser.parse_args()

# #     texts = []
# #     for r in load_jsonl(args.input_file):
# #         texts.append(r['text'])

# #     assemble_events(texts, args.model_ner, args.model_event, args.output_file)


# ### FILE: NOTES.md
# """
# Limitations and next steps:
# - This baseline splits NER and event role extraction. Joint models (e.g. seq2seq T5 that directly outputs JSON) can do better.
# - Domain adaptation: continue pretraining on financial text (MLM) yields measurable gains.
# - Use weak supervision to harvest more labeled examples from press release templates and 8-K filings.
# - Use entity linking to canonicalize company names to tickers.
# - For money/date normalization use rule-based post-processing and currency conversion.

# """

import argparse
from datasets import ClassLabel
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers import AutoTokenizer
import numpy as np

from dataset import load_ner_dataset


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            new_labels.append(label)
    return new_labels


def convert_bio_to_ids(label_list):
    unique = sorted(set(label_list))
    id_map = {l:i for i,l in enumerate(unique)}
    return unique, id_map


def train(args):
    ds, tokenizer, labels = load_ner_dataset(args.train_file, args.model_name)

    flat = set()
    for ll in ds['labels']:
        for x in ll:
            flat.add(x)
    unique_labels = sorted(flat)
    label2id = {l:i for i,l in enumerate(unique_labels)}
    id2label = {i:l for l,i in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, return_tensors=None)
        labels_aligned = []
        for i, labels_seq in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(labels_seq, word_ids)
            aligned_ids = [label2id[l] if (l != -100 and l in label2id) else -100 if l== -100 else label2id.get(l, -100) for l in aligned]
            labels_aligned.append(aligned_ids)
        tokenized_inputs['labels'] = labels_aligned
        return tokenized_inputs

    tokenized = ds.map(tokenize_and_align, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load('seqeval')

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in lab if l != -100] for lab in labels]
        true_preds = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            'precision': results['overall_precision'],
            'recall': results['overall_recall'],
            'f1': results['overall_f1']
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy='epoch',
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--output_dir', type=str, default='./ner_model')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train(args)