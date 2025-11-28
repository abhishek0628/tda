"""
Dataset utilities using Hugging Face datasets and tokenizers.
"""
import os
from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer
from utils import load_jsonl, charspan_to_tokenspan


def load_ner_dataset(jsonl_path: str, model_name: str, label_list: List[str]=None):
    # Produces a HuggingFace Dataset with fields: tokens, labels (BIO)
    raw = list(load_jsonl(jsonl_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    examples = {
        'tokens': [],
        'labels': []
    }

    # Build label mapping if not provided
    labels_set = set()
    for r in raw:
        for ent in r.get('entities', []):
            labels_set.add(ent['type'])
    labels = sorted(labels_set) if label_list is None else label_list
    label_to_id = {l:i for i,l in enumerate(labels)}

    for r in raw:
        txt = r['text']
        enc = tokenizer(txt, truncation=True, return_offsets_mapping=True)
        offsets = enc['offset_mapping']
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
        token_labels = ['O'] * len(tokens)

        for ent in r.get('entities', []):
            # map char span to token span
            start, end = ent['start'], ent['end']
            ts, te = charspan_to_tokenspan(start, end, offsets)
            if ts is None or te is None:
                continue
            token_labels[ts] = 'B-' + ent['type']
            for idx in range(ts+1, te+1):
                token_labels[idx] = 'I-' + ent['type']

        examples['tokens'].append(tokens)
        examples['labels'].append(token_labels)

    ds = Dataset.from_dict(examples)
    return ds, tokenizer, labels