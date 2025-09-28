import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import time
import evaluate

# Use finbert, a transformer pretrained on finance sentiment data, perfect for our purposes
MODEL_NAME = "yiyanghkust/finbert-tone"
FINETUNED_MODEL_NAME = "finbert_finetuned"

w_negative = None
w_neutral = None
w_positive = None

# Weighted Trainer class, so loss is computed inversely proportional to the ground truth label ratio
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # weights inversely proportional to class frequency
        class_weights = torch.tensor([w_negative, w_neutral, w_positive], dtype=torch.float32).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def loadDataset():
    global w_negative, w_neutral, w_positive
    raw_data = pd.read_parquet('../data/parquet/labeled_sentiments.parquet', engine='fastparquet')

    raw_data['sentiment'] = raw_data['sentiment'].map({"positive": 2, "neutral": 1, "negative": 0})
    raw_data = raw_data.rename({'sentiment': 'labels'}, axis=1)
    raw_data = raw_data.drop('length', axis=1)


    counts = raw_data['labels'].value_counts()

    totals_df = pd.DataFrame({num: [counts.sum()] for num in range(3)})
    mean_weights = pd.DataFrame({num: (totals_df / counts).mean(axis=1) for num in range(3)})

    weights = (totals_df / counts) / (mean_weights)# Normalize so weights average out to 1
    
    w_negative = weights[0].iloc[0]
    w_neutral = weights[1].iloc[0]
    w_positive = weights[2].iloc[0]


    ds = Dataset.from_pandas(raw_data)
    ds = ds.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    encoded = ds.map(tokenize, batched=True)

    return encoded, tokenizer



def loadModel():
    device = torch.device('mps') if torch.mps.is_available() else torch.device('cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    
    model.to(device)

    return model
    

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

training_args = TrainingArguments(
    output_dir='./logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True
)


encoded, tokenizer = loadDataset()
trainer = WeightedTrainer(
    model=loadModel(),
    args=training_args,
    train_dataset=encoded['train'],
    eval_dataset=encoded['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.evaluate()

trainer.save_model(FINETUNED_MODEL_NAME)
tokenizer.save_pretrained(FINETUNED_MODEL_NAME)