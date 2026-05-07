# Train on Kaggle and link models to this app

This project **does not** ship a training script; it emits labelled JSON Lines and expects you to drop **Hugging Face–compatible** checkpoints into two folders after you fine-tune **DistilBERT** (or another base) elsewhere—e.g. on Kaggle.

## 1. What to copy from this app before Kaggle

From your machine, take the datasets the app generates when you finalize documents:

| Local path | Purpose |
|------------|---------|
| `data/training/train.jsonl` | Training rows |
| `data/training/validation.jsonl` | Validation rows (may be sparse or empty initially) |

Each line is a JSON object with (among others):

- **`text`** — OCR / extracted PDF text  
- **`label_main`** or **`main_bucket`** — main class  
- **`label_sub`** or **`sub_bucket`** — sub-class  
- **`main_keywords`** — optional; useful to append as extra context during training  

> **Git note:** `.gitignore` ignores `data/` and `*.jsonl`. Copy files manually into a zip or upload them as a **Kaggle Dataset**.

## 2. Label strings must match the app exactly

Inference checks returned labels against these tuples (`app/services/classify.py`). Your saved checkpoints must expose **the same human-readable labels** (`id2label` in `config.json`).

While the Flask app runs, **`GET /api/taxonomy`** returns the authoritative main/sub lists as JSON (same source the SPA uses).

**Main buckets (2-way unless you extend the code):**

- `Transactional Documents`  
- `Certifications`  

Training data stored as **`Transactional`** is canonicalised here to **`Transactional Documents`** when writing JSONL in some flows; keeping **`Transactional Documents`** everywhere avoids surprises.

**Sub buckets:**

- Transactional: `Invoice`, `Purchase Order`, `Proof of Payment`, `Production Records`, `Transportation Records`  
- Certifications: `Social Audit Reports`, `Security Audit Records`, `Third party certification`  

You need **fine-tuned models**:

1. **Main model** — `num_labels = len(main buckets)` in your union (usually 2).  
2. **Sub model** — `num_labels = 10` (all sub-classes above).

## 3. Recommended base model on Kaggle

Use the same family the app historically targeted:

```
distilbert-base-multilingual-cased
```

Fine-tune with Hugging Face `AutoModelForSequenceClassification` so the export looks like any standard `Trainer.save_pretrained(...)` checkpoint (tokenizer + weights + config).

Optional: match how JSONL may be richer by feeding the classifier:

```
text_for_model = row["text"] + "\nKEYWORDS: " + " ".join(row.get("main_keywords") or [])
```

Truncate (e.g. 512 tokens) like `classify.py` truncate at inference (`text[:2048]` chars—align tokenizer limit with training).

## 4. Train two checkpoints on Kaggle

Typical layout after training saves:

```
main_bucket_classifier/
  config.json  tokenizer.json tokenizer_config.json  model.safetensors (or pytorch_model.bin)  ...

sub_bucket_classifier/
  config.json ...
```

Ensure each `config.json` has **`id2label`** / **`label2id`** whose **values are the exact strings** listed in section 2.

## 5. Link the models to this app

1. On your laptop (or deployment server), clone/copy the repo.  
2. Create or replace these directories **relative to the project root** (same folder as `run.py`):

   ```
   models/main_bucket_classifier/   ← main task checkpoint contents
   models/sub_bucket_classifier/    ← sub task checkpoint contents
   ```

3. Paste **all** tokenizer + model files from Kaggle downloads into those folders respectively.  
4. Run the Flask app **from the project root** so `Path("models/...")` resolves:

   ```bash
   python run.py
   ```

When both folders exist and load cleanly, `app/services/classify.py` uses **`transformers.pipeline("text-classification", ...)`** on them. If loading fails or folders are missing, it falls back to heuristics.

5. Turn on GPU inference later by changing **`device=-1`** in `_Predictor._load_pipeline` if you deploy on CUDA (optional).

## 6. Sanity check

- Minimum data: meaningful training needs enough examples **per class**, especially **both main buckets** and **several sub-types**.  
- After swapping models, process a PDF you’ve never trained on and confirm main/sub labels match expectations.  

## 7. Repo vs large files

`.gitignore` includes `models/` and `*.bin` so checkpoints stay **out of git**. Distribute artefacts via drives, artefact buckets, or your deploy pipeline—not the repository.
