### FILE: utils.py
"""
Helper utilities for tokenization, span conversion, and IO.
"""
import json
from typing import List, Tuple, Dict

from dateutil import parser as dateparser


def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(path: str, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_date(text: str):
    try:
        dt = dateparser.parse(text, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return text


def charspan_to_tokenspan(char_start: int, char_end: int, token_offsets: List[Tuple[int,int]]):
    # token_offsets is list of (start_char, end_char) for each token
    ts, te = None, None
    for i,(s,e) in enumerate(token_offsets):
        if ts is None and char_start >= s and char_start < e:
            ts = i
        if te is None and char_end > s and char_end <= e:
            te = i
    # fallback: find nearest
    if ts is None:
        for i,(s,e) in enumerate(token_offsets):
            if s >= char_start:
                ts = i
                break
    if te is None:
        for i,(s,e) in enumerate(reversed(token_offsets)):
            if e <= char_end:
                te = len(token_offsets)-1 - i
                break
    return ts, te

