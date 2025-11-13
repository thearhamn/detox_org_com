import torch
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

def main():
    MODEL_ID = "distilroberta-base"

    # Load the CSV files
    train_df = pd.read_csv('train_data.csv')
    val_df = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Rename 'tweet' column to 'text' for consistency
    train_dataset = train_dataset.rename_column('tweet', 'text')
    val_dataset = val_dataset.rename_column('tweet', 'text')
    test_dataset = test_dataset.rename_column('tweet', 'text')

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    # Set dataset format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "class"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "class"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "class"])

    # Define labels
    num_labels = 2  # Binary classification: 0 (not offensive) and 1 (offensive)
    id2label = {0: "not_offensive", 1: "offensive"}
    label2id = {"not_offensive": 0, "offensive": 1}

    print(f"number of labels: {num_labels}")
    print(f"the labels: {list(id2label.values())}")

    # Update the model's configuration
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.update({
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id
    })

    # Metrics
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="model/logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard",
    )

    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(MODEL_ID, config=config)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model
    print("\nValidation set evaluation:")
    val_results = trainer.evaluate()
    print(val_results)

    print("\nTest set evaluation:")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(test_results)

    # Save the fine-tuned model
    print("\nSaving model...")
    trainer.save_model("model/pretrained_DistilRoberta")
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 