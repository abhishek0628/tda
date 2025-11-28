
### FILE: inference.py
"""
End-to-end inference: run NER to get candidate spans, then run event classifier to assign roles and assemble events.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from utils import load_jsonl, save_jsonl, normalize_date


def extract_spans_ner(ner_model_dir, texts):
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(ner_model_dir, use_fast=True)
    nlp = pipeline('ner', model=ner_model, tokenizer=tokenizer, aggregation_strategy='simple')
    all_spans = []
    for t in texts:
        preds = nlp(t)
        # preds include {'entity_group','score','word','start','end'}
        spans = [{'type': p['entity_group'], 'text': p['word'], 'start': p['start'], 'end': p['end'], 'score': p['score']} for p in preds]
        all_spans.append(spans)
    return all_spans


def run_event_classifier(event_model_dir, tokenizer, trigger_text, arg_text, sentence):
    model = AutoModelForSequenceClassification.from_pretrained(event_model_dir)
    # tokenizer passed in from event model
    combined = f"[TRG] {trigger_text} [ARG] {arg_text} [SENT] {sentence}"
    inputs = tokenizer(combined, truncation=True, return_tensors='pt')
    logits = model(**inputs).logits
    probs = logits.softmax(-1).detach().cpu().numpy()[0]
    label_id = probs.argmax()
    # label string mapping may not be saved; assume user saved labels separately in production.
    return label_id, float(probs[label_id])


def assemble_events(texts, ner_model_dir, event_model_dir, output_file):
    texts_list = [t for t in texts]
    spans_list = extract_spans_ner(ner_model_dir, texts_list)
    # load tokenizer used for event classifier
    tokenizer = AutoTokenizer.from_pretrained(event_model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(event_model_dir)

    results = []
    for text, spans in zip(texts_list, spans_list):
        # naive trigger selection: verbs or EVENT labels; otherwise use all spans as triggers
        triggers = [s for s in spans if s['type'] in ('EVENT', 'VERB')]
        if not triggers:
            triggers = spans

        events_for_text = []
        for trg in triggers:
            args_candidates = spans
            arguments = []
            for arg in args_candidates:
                combined = f"[TRG] {trg['text']} [ARG] {arg['text']} [SENT] {text}"
                inputs = tokenizer(combined, truncation=True, return_tensors='pt')
                logits = model(**inputs).logits
                probs = logits.softmax(-1).detach().cpu().numpy()[0]
                label_id = int(probs.argmax())
                score = float(probs[label_id])
                if score > 0.5:  # threshold; in practice tune
                    arguments.append({'role_id': label_id, 'role_confidence': score, 'text': arg['text'], 'span': [arg['start'], arg['end'], arg['type']]})
            if arguments:
                events_for_text.append({'trigger': trg['text'], 'trigger_span': [trg['start'], trg['end']], 'arguments': arguments})
        results.append({'text': text, 'events': events_for_text})

    save_jsonl(output_file, results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ner', type=str, required=True)
    parser.add_argument('--model_event', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='results.jsonl')
    args = parser.parse_args()

    texts = []
    for r in load_jsonl(args.input_file):
        texts.append(r['text'])

    assemble_events(texts, args.model_ner, args.model_event, args.output_file)

