import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from hydra import compose, initialize
from omegaconf import DictConfig

from src.train.intent_dataset import IntentDataset


def train(cfg: DictConfig):
    """Train intent classification model"""
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, num_labels=cfg.num_intents
    )

    # Prepare datasets
    train_dataset = IntentDataset(Path(cfg.data.train_path))
    val_dataset = IntentDataset(Path(cfg.data.val_path))

    # Data preprocessing function
    def preprocess_function(examples):
        texts, labels = zip(*examples)
        encoding = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Need to convert labels to numeric indices here
        # In actual projects, may need to build label mapping
        label_map = {"unknown": 0, "literal": 1, "metaphorical": 2}  # Example label mapping
        encoding["labels"] = [label_map.get(label, 0) for label in labels]
        return encoding

    # Prepare DataLoader
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Need to convert labels to numeric indices here
        label_map = {"unknown": 0, "literal": 1, "metaphorical": 2}  # Example label mapping
        encoding["labels"] = torch.tensor([label_map.get(label, 0) for label in labels])
        return encoding

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.max_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

    # Save best model
    best_model_path = Path(cfg.output_dir) / "best.ckpt"
    trainer.save_model(best_model_path)
    
    print(f"Training completed, best model saved to {best_model_path}")


if __name__ == "__main__":
    # Initialize Hydra configuration
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="intent_cls")
        
        # Ensure output directory exists
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Start training
        train(cfg)