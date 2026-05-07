# PDF Classification & Extraction Platform (Flask)

A clean Flask starter architecture for a PDF intelligence pipeline based on your flowchart:

- PDF upload endpoint
- Pre-processing (PyMuPDF + EasyOCR fallback placeholder)
- Main bucket classifier (loads externally trained checkpoints; heuristics fallback)
- Sub-bucket classifier
- Key info extractor (spaCy NER + regex placeholder)
- Storage/grouping engine
- Polished web UI

## Project Structure

```
my_second_python/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   └── document.py
│   ├── routes/
│   │   └── main.py
│   ├── services/
│   │   ├── classify.py
│   │   ├── extract.py
│   │   ├── grouping.py
│   │   ├── mlops.py
│   │   └── preprocess.py
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/
│       ├── base.html
│       ├── index.html
│       └── result.html
├── data/
│   ├── processed/
│   └── uploads/
├── requirements.txt
└── run.py
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## External training (e.g. Kaggle)

Fine-tune `distilbert-base-multilingual-cased` on **`data/training/train.jsonl`** / **`validation.jsonl`**, then place checkpoints under **`models/main_bucket_classifier/`** and **`models/sub_bucket_classifier/`**. See **[docs/kaggle-training.md](docs/kaggle-training.md)** for labels, layout, and wiring.

## Next Up (Production)

- Plug real models into `app/services/classify.py` (ONNXRuntime INT8)
- Add PyMuPDF + EasyOCR logic in `app/services/preprocess.py`
- Replace in-memory grouping with SQLite persistence
- Add feedback endpoint to send correction events to retraining queue

