import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--model-path", default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    args = parser.parse_args()


    max_steps = args.max_steps if args.max_steps is not None else -1

    model_name = args.model_path.rstrip("/").split("/")[-1].replace(".", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{ts}"

    set_seed(SEED)

    with open(f"{args.dataset_path}/label2id.json", encoding="utf-8") as f:
        label2id = json.load(f)
        
    with open(f"{args.dataset_path}/id2label.json", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}


    ds_train = load_dataset("json", data_dir=args.dataset_path, split="train")
    ds_val = load_dataset("json", data_dir=args.dataset_path, split="validation")
    split = DatasetDict(train=ds_train, validation=ds_val)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    split = split.map(tokenize, batched=True, remove_columns=["text"])
    split = split.rename_column("label", "labels")
    split.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, pred),
            "f1": f1_score(labels, pred, average="weighted"),
        }

    out_dir = str(args.output_dir)
    train_args = TrainingArguments(
        output_dir=out_dir,
        run_name=run_name,
        seed=SEED,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=split["train"],
        eval_dataset=split["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
