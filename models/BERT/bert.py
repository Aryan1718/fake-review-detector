import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets", "fake reviews dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

def get_model_path():
    return MODEL_DIR

def train_model():
    print("--- Training BERT Model ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).fillna('')
    # Ensure labels are integers
    df['label'] = df['label'].astype(int)

    # Split (Smaller subset for demo training if needed, but full for real)
    # Using 10% for speed if user just wants to test flow? No, keep logic correct.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "results"),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        no_cuda=False if torch.cuda.is_available() else True # Use CPU if no CUDA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting Training...")
    trainer.train()

    # Save
    print(f"Saving model to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Done.")

def load_and_predict(text):
    if not os.path.exists(MODEL_DIR):
        return None

    try:
        # Load (Lazy loading could be better, but for simplicity load here)
        # Note: In a real app we might cache the loaded model globaly in app.py
        # But for this interface, we return the prediction logic.
        # This function might re-load model every time if called directly, 
        # so app.py should handle caching.
        
        # We will assume app.py handles the loading state, but here is the logic:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        label_map = {0: "Human-Written", 1: "AI-Generated"}
        return label_map.get(prediction, "Unknown")
        
    except Exception as e:
        print(f"Error in BERT prediction: {e}")
        return None

if __name__ == "__main__":
    train_model()
