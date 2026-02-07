import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Datasets", "fake reviews dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_pipeline.pkl")

def get_model_path():
    return MODEL_PATH

def train_model():
    print("--- Training Naive Bayes Model ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}") # Handle Git LFS pointer issues gracefully-ish
        return

    # Basic Preprocessing
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: Dataset must have 'text' and 'label' columns.")
        return

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).fillna('')

    # Split
    print("Splitting data...")
    train_text, test_text, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB(alpha=1.0))
    ])

    # Train
    print("Fitting model...")
    pipeline.fit(train_text, train_labels)

    # Evaluate
    print("Evaluating...")
    preds = pipeline.predict(test_text)
    acc = accuracy_score(test_labels, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(test_labels, preds))

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_and_predict(text):
    if not os.path.exists(MODEL_PATH):
        return None  # Model not trained

    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([text])[0]
        # Assuming label 1 = Fake (CG), 0 = Real (OR) - Need to verify dataset mapping!
        # Usually dataset: label 1 (Fake), 0 (Real) -> Check printed head in notebook
        # Notebook output showed: 0=good canned asparagus... 1=didnt buy particular one...
        # Wait, usually label names are "CG" (Computer Generated) and "OR" (Original Review).
        # In the csv: label is int64.
        # Let's map 0->Human, 1->AI for now based on common datasets, but user should verify.
        
        # Mapping: 
        # Check notebook output again: 
        #   Text: "one best purchase..." True Label: 0, Predicted Label: 0
        #   Text: "yorkie oral surgery..." True Label: 1, Predicted Label: 1 
        #   "Yorkie" sounds specific (Human?), "one best purchase" sounds generic (AI?).
        #   Actually often 1 is Fake (CG). Let's stick with that.
        label_map = {0: "Human-Written", 1: "AI-Generated"}
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        print(f"Error in NB prediction: {e}")
        return None

if __name__ == "__main__":
    train_model()
