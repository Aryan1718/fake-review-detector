import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets", "fake reviews dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model") # Changed from saved_tinyllama_model for consistency

def get_model_path():
    return MODEL_DIR

def train_model():
    print("--- Training TinyLlama Model ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Load and preprocess the dataset
    try:
        data = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    data = data[['label', 'text']].fillna('')  # Ensure no missing values in text or labels
    
    # Split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        data['text'], data['label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    # Convert datasets to Hugging Face's Dataset format
    train_data = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
    val_data = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})
    
    # Load TinyLlama tokenizer and model
    model_name = "TinyLlama/TinyLlama_v1.1" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=64
        )
    
    train_data = train_data.map(tokenize_function, batched=True).rename_column("label", "labels").remove_columns(["text"])
    val_data = val_data.map(tokenize_function, batched=True).rename_column("label", "labels").remove_columns(["text"])
    
    train_data.set_format("torch")
    val_data.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "results"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        no_cuda=False if torch.cuda.is_available() else True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

def load_and_predict(text):
    if not os.path.exists(MODEL_DIR):
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            
        label_map = {0: "Human-Written", 1: "AI-Generated"}
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        print(f"Error in TinyLlama prediction: {e}")
        return None

if __name__ == "__main__":
    train_model()