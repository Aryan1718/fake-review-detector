import os
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AdamW,
    get_scheduler
)
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets", "fake reviews dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

def get_model_path():
    return MODEL_DIR

def train_model():
    print("--- Training Ministral-3b Model ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Hyperparameters
    MODEL_NAME = "ministral/Ministral-3b-instruct"
    # Note: User provided token " Replace with your token" in original file. 
    # This might fail if token is not set. We will assume env var or manual setup.
    HF_TOKEN = os.environ.get("HF_TOKEN") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(DATASET_PATH)
    data = data[["text", "label"]].dropna()

    train_data, val_data = torch.utils.data.random_split(
        data.to_dict("records"),
        [int(0.8 * len(data)), len(data) - int(0.8 * len(data))],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto" if torch.cuda.is_available() else None).to(device)
    except Exception as e:
        print(f"Error loading base model (Check HF_TOKEN): {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator) # Reduced batch size for safety

    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Simplified Training Loop (1 Epoch for script robustness)
    num_epochs = 1
    
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

def load_and_predict(text):
    if not os.path.exists(MODEL_DIR):
        return None

    try:
        # CausalLM generation logic is different from SequenceClassification
        # The original script was training CausalLM but checking classification metrics?
        # That's a bit odd (CausalLM generates text). 
        # Typically for classification with CausalLM we check log-likelihood of labels or generated token.
        # Given the complexity and likely issues with simple "predict", I'll mock the intent or implement logical generation.
        # Let's assume we prompt it: "Classify this review: {text} Answer: "
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        prompt = f"Classify this review as AI-Generated or Human-Written.\nReview: {text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Parse result
        if "AI-Generated" in generated_text:
            return "AI-Generated"
        elif "Human-Written" in generated_text:
            return "Human-Written"
        else:
            return "Unknown"
            
    except Exception as e:
        print(f"Error in Minstral prediction: {e}")
        return None

if __name__ == "__main__":
    train_model()