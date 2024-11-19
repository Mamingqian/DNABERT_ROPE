import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from datasets import Dataset
from sklearn.metrics import matthews_corrcoef, accuracy_score
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    AdamW,
    set_seed,
    RoFormerConfig,
    RoFormerForMaskedLM
)

# Set environment paths
os.environ["AMLT_DATA_DIR"] = "../data"
os.environ["AMLT_OUTPUT_DIR"] = "../output"

@dataclass
class ModelArguments:
    model_name_or_path: str = "zhihan1996/DNABERT-2-117M"

@dataclass
class DataTrainingArguments:
    dataset_name: str = os.environ.get("AMLT_DATA_DIR")
    max_seq_length: int = 128

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"mcc": mcc, "accuracy": acc, "length": len(labels)}

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)  # Set seed for transformers

    if dist.is_available() and dist.is_initialized():
        seed += dist.get_rank()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

class SequenceDataset(IterableDataset):
    def __init__(self, data_file, tokenizer, max_seq_length):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self):
        with open(self.data_file, "r") as file:
            for line in file:
                sequence = line.strip()
                if sequence:
                    tokenized = self.tokenizer(
                        sequence,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_tensors="pt"
                    )
                    yield {
                        "input_ids": tokenized["input_ids"].squeeze(0),
                        "attention_mask": tokenized["attention_mask"].squeeze(0),
                    }

def main():

    set_all_seeds(42)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Model arguments: {model_args}")
    print(f"Data arguments: {data_args}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print("Loaded tokenizer and data collator")

    config = RoFormerConfig.from_pretrained(model_args.model_name_or_path)
    model = RoFormerForMaskedLM(config)


    train_path, valid_path = os.path.join(data_args.dataset_name, "train.txt"), os.path.join(data_args.dataset_name, "dev.txt")
    
    train_dataset = SequenceDataset(train_path, tokenizer, data_args.max_seq_length)
    valid_dataset = SequenceDataset(valid_path, tokenizer, data_args.max_seq_length)
    print("Initialized datasets")


    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-5
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=30000,
        num_training_steps=500000
    )

    training_args = TrainingArguments(
        output_dir=os.environ.get("AMLT_OUTPUT_DIR"),
        overwrite_output_dir=True,
        per_device_train_batch_size=512,
        gradient_accumulation_steps=2,
        max_steps=500000,
        logging_steps=500,
        save_steps=10000,
        evaluation_strategy="steps",
        eval_steps=10000,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train()

if __name__ == "__main__":
    main()
