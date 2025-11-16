# Offensive‑Language Detoxification and Classification Toolkit

This repository contains code and data for building a two‑stage system that detects offensive social media posts, rewrites them to be less toxic, and evaluates the quality of the rewrites. The workflow is designed around three main steps:

- **Data preparation** – download and split a labelled dataset into training/validation/test CSV files.
- **Detoxification** – run a sequence‑to‑sequence model over the test set to produce a "detoxified" version of each tweet.
- **Classification and evaluation** – fine‑tune a sequence classification model to recognise offensive language and apply it to both the original and detoxified tweets. Additional metrics such as semantic similarity and BERTScore are computed to assess how well the detoxified texts retain meaning while reducing toxicity.

## Quick Start

The suggested execution order is as follows:

1. Run `preprocess_data.ipynb` to create `train_data.csv`, `val_data.csv` and `test_data.csv` from the `tdavidson/hate_speech_offensive` dataset and to convert the original three‑class labels to a binary offensive/not‑offensive scheme raw.githubusercontent.com.

2. Run `process_test_data.py` (or the equivalent notebook `detoxify.ipynb`) to generate `test_data_detoxified.csv`. This script loads a multilingual detoxification model (`s‑nlp/mt0-xl-detox-orpo`) and rewrites each test tweet, preserving the original semantics whenever possible raw.githubusercontent.com.

3. Train a classifier. You may choose one of the provided notebooks:
   - `classifier_distilbert.ipynb` – fine‑tunes `distilbert-base-uncased` on the prepared CSV files. It tokenizes the tweet text, maps the labels to the labels column, defines appropriate training arguments, and saves the best model raw.githubusercontent.com.
   - `classifier_distilroberta.ipynb` – fine‑tunes `distilroberta-base` with a similar setup raw.githubusercontent.com.
   - `classifier.py` – a Python script version of the DistilRoBERTa training loop. It reads the CSVs, renames the tweet column to text, tokenizes the data and creates a Trainer object. The model, training arguments, and metrics are defined in the script before launching fine‑tuning and saving the resulting weights raw.githubusercontent.com.

4. Run `evaluation.ipynb` to assess the detoxified outputs. This notebook loads the fine‑tuned classifier, applies it to the detoxified text via the HuggingFace pipeline API raw.githubusercontent.com, computes semantic similarity using a sentence‑embedding model (`nomic‑ai/nomic‑embed‑text‑v1.5`), and uses BERTScore to evaluate lexical preservation raw.githubusercontent.com.

## Repository Structure

The following table gives a brief overview of the key files. Each entry contains only a short description; more details are provided in the sections above.

| File or folder | Purpose |
|---------------|---------|
| `preprocess_data.ipynb` | Download the `tdavidson/hate_speech_offensive` dataset, convert labels to binary (0 = non‑offensive, 1 = offensive) and split into training/validation/test CSVs raw.githubusercontent.com. |
| `process_test_data.py` | Command‑line script to detoxify `test_data.csv` using the `s‑nlp/mt0-xl-detox-orpo` model and write `test_data_detoxified.csv` raw.githubusercontent.com. |
| `detoxify.ipynb` | Notebook version of the detoxification script with helper functions and progress bars raw.githubusercontent.com. |
| `classifier_distilbert.ipynb` | Fine‑tune a DistilBERT classifier for offensive language detection raw.githubusercontent.com. |
| `classifier_distilroberta.ipynb` | Fine‑tune a DistilRoBERTa classifier for offensive language detection raw.githubusercontent.com. |
| `classifier.py` | Pure‑Python implementation of the DistilRoBERTa training loop for command‑line use raw.githubusercontent.com. |
| `evaluation.ipynb` | Evaluate detoxified outputs using semantic similarity, classifier predictions and BERTScore raw.githubusercontent.com raw.githubusercontent.com. |
| `train_data.csv`, `val_data.csv`, `test_data.csv` | CSV files produced by the preprocessing step. |
| `test_data_detoxified.csv` | Detoxified test data produced by `process_test_data.py` or `detoxify.ipynb` raw.githubusercontent.com. |
| `requirements.txt` | List of Python package dependencies raw.githubusercontent.com. |
| `LICENSE` | MIT licence for the project. |

## Installation

It is recommended to use a virtual environment. To install the dependencies, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models used for detoxification (`s‑nlp/mt0-xl-detox-orpo`) and classification (`distilbert-base-uncased` or `distilroberta-base`) are downloaded automatically via the HuggingFace Transformers library when the scripts are executed.

## Usage

### Prepare the data

Launch `preprocess_data.ipynb` in Jupyter or convert it to a script and run it. This will create `train_data.csv`, `val_data.csv` and `test_data.csv` in the current directory.

### Detoxify the test set

Run `python process_test_data.py` to create `test_data_detoxified.csv`. If you prefer a notebook, open `detoxify.ipynb` instead. Both versions load the test CSV, process the texts in batches using the detoxification model, and save the results raw.githubusercontent.com.

### Train a classifier

- **For DistilBERT**: open `classifier_distilbert.ipynb` and execute the cells. At the end of training the notebook saves `model/pretrained_DistilBERT`.
- **For DistilRoBERTa**: either open `classifier_distilroberta.ipynb` or run `python classifier.py` from the command line. Both approaches fine‑tune the model on the prepared CSVs and save the weights to `model/pretrained_DistilRoberta` raw.githubusercontent.com.

### Evaluate

Open `evaluation.ipynb` and adjust `MODEL_PATH` to point to the folder created in the previous step (e.g., `model/pretrained_DistilRoberta`). Execute the notebook to compute similarity, classification outcomes and BERTScore metrics raw.githubusercontent.com raw.githubusercontent.com. The notebook summarises how the detoxified texts compare to the originals in terms of toxicity reduction and meaning preservation.

## Notes

- GPU acceleration is highly recommended for both detoxification and model training due to the size of the underlying transformer models.

- The detoxification model is multilingual and expects an appropriate language prompt (e.g., "Detoxify: " for English) to be prepended to each input. This is handled for you in the provided script and notebook raw.githubusercontent.com.

- If you wish to experiment with different detoxification models or classification architectures, modify the corresponding `MODEL_ID` in the scripts or notebooks.

By following the steps outlined above, you can reproduce the full pipeline—from dataset preparation through detoxification, fine‑tuning and evaluation—and extend it to new models or datasets.
