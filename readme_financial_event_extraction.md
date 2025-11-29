# Financial Event Extraction System

This project provides a Transformer-based system to extract and structure financial events from textual news data. It uses pre-trained NER and classification models to identify entities, event types, dates, and monetary values from financial news.

---

## Features

- **Event Classification**: Predicts the type of financial event (e.g., MERGER, ACQUISITION, EARNINGS, PARTNERSHIP).
- **Named Entity Recognition (NER)**: Extracts entities such as company names, monetary values, and dates.
- **Structured Output**: Converts unstructured text into a JSON format for easy integration with analytics pipelines.
- **Lightweight**: Designed to run with minimal dependencies.

---

## Requirements

- Python 3.10 or higher
- PyTorch
- Transformers
- python-dateutil

clone the project
```bash
mkdir project && cd project
git clone https://github.com/Abhishek0628/tda.git
cd tda
```

Install dependencies using pip:

```bash
python3 -m venv venv
source venv/bin/activate
pip install transformers datasets evaluate seqeval torch accelerate tokenizers python-dateutil
```

---

```bash
python financial_event_infer.py --task toy
python financial_event_infer.py --task ner --train --model_name_or_path bert-base-uncased --output_dir ./models/ner
python financial_event_infer.py --task cls --train --model_name_or_path roberta-base --output_dir ./models/cls
```
make file input.txt and paste for example
```bash
TechNova Inc. announced a partnership with GreenEnergy Ltd. on 15th December 2025 to develop renewable energy solutions. The deal is valued at $250 million.
```


## Project Structure

```
financial_event_extraction/
├── models/
│   ├── ner/       # Pre-trained NER model
│   └── cls/       # Pre-trained classification model
├── financial_event_infer.py   # Main inference script
├── input.txt      # Example input text file
└── README.md
```

---

## Usage

### Inference

Run the script with a text file containing financial news:

<!-- ```bash
python3 financial_event_infer.py input.txt
``` -->
Run the file
```bash
python financial_event_infer.py \
  --task infer \
  --ner_model ./models/ner \
  --cls_model ./models/cls \
  --text_file input.txt
```

**Example `input.txt`:**

```
TechNova Inc. announced a partnership with GreenEnergy Ltd. on 15th December 2025 to develop renewable energy solutions. The deal is valued at $250 million.
```

**Expected output:**

```json
{
  "text": "TechNova Inc. announced a partnership with GreenEnergy Ltd. on 15th December 2025 to develop renewable energy solutions. The deal is valued at $250 million.",
  "event": {
    "label": "PARTNERSHIP",
    "score": 0.95
  },
  "entities": [
    {"type": "COMPANY", "text": "TechNova Inc."},
    {"type": "COMPANY", "text": "GreenEnergy Ltd."}
  ],
  "money_mentions": ["$250 million"],
  "date_mentions": ["15th December 2025"]
}
```

---

## How It Works

1. **Load Models**: Loads pre-trained NER and classification models from `./models/ner` and `./models/cls`.
2. **NER Prediction**: Tokenizes the input text and predicts entity labels for each token.
3. **Classification Prediction**: Classifies the overall event type in the text.
4. **Post-processing**: Extracts structured entities, dates, and monetary amounts.
5. **Output**: Returns a JSON object containing all extracted information.

---

## Notes

- Ensure your models are trained or fine-tuned before running inference.
- The script supports only one input file at a time.
- Monetary and date extraction uses regex and may need fine-tuning for complex formats.

---

## License

MIT License

