
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# ✅ Load data
train_df = pd.read_csv("Data/train/input/input_train_annot_t5.csv").dropna()
val_df = pd.read_csv("Data/val/input/input_val_annot_t5.csv").dropna()
test_df = pd.read_csv("Data/test/input/input_test_annot_t5.csv").dropna()

# ✅ Dataset Class
class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=512, max_target_len=64):
        self.tokenizer = tokenizer
        self.inputs = df["input"].tolist()
        self.targets = df["output"].tolist()
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx], 
            max_length=self.max_input_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

# ✅ Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ✅ Create datasets
train_dataset = QADataset(train_df, tokenizer)
val_dataset = QADataset(val_df, tokenizer)
test_dataset = QADataset(test_df, tokenizer)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./t5-model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    #load_best_model_at_end=True,
    report_to="none"
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ✅ Train the model
trainer.train()

# ✅ Evaluate on test set
def evaluate(model, dataset, tokenizer):
    model.eval()
    preds, targets = [], []
    for sample in dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)[0]
        pred = tokenizer.decode(output_ids, skip_special_tokens=True)
        target = tokenizer.decode(sample["labels"][sample["labels"] != -100], skip_special_tokens=True)

        preds.append(pred.strip())
        targets.append(target.strip())

    # Print accuracy or example outputs
    for i in range(min(10, len(preds))):
        print(f"Q: {test_df['Input'].iloc[i]}\nPred: {preds[i]}\nTrue: {targets[i]}\n")

evaluate(model, test_dataset, tokenizer)
