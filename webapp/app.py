import os
import sys
import torch
from flask import Flask, render_template, request, jsonify
from collections import Counter

# Add parent directory to sys.path to import models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

# Import model interfaces
# We wrap imports in try-except to avoid crashing if dependencies are missing or scripts have errors
tinyllama = None
bert = None
naive_bayes = None
ministral = None

try:
    from models.TinyLlama import tinyllama
except ImportError as e:
    print(f"Warning: Could not import TinyLlama script (dependencies missing?): {e}")

try:
    from models.BERT import bert
except ImportError as e:
    print(f"Warning: Could not import BERT script (dependencies missing?): {e}")

try:
    from models.Naive_Bayes import naive_bayes
except ImportError as e:
    print(f"Warning: Could not import Naive Bayes script (dependencies missing?): {e}")

try:
    from models.Ministral_3b_instruct import ministral_3b_instruct as ministral
except ImportError as e:
    print(f"Warning: Could not import Ministral script (dependencies missing?): {e}")

app = Flask(__name__)

class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Checks which models are trained and registers them."""
        print("--- Loading Ensemble Models ---")
        
        # Define available model modules
        potential_models = {
            "TinyLlama": tinyllama,
            "BERT": bert,
            "Naive Bayes": naive_bayes,
            "Ministral": ministral
        }

        for name, module in potential_models.items():
            if module is None:
                continue
                
            try:
                model_path = module.get_model_path()
                if os.path.exists(model_path):
                    print(f"[{name}] Found trained model at {model_path}")
                    self.models[name] = module
                else:
                    print(f"[{name}] No trained model found at {model_path}")
            except Exception as e:
                print(f"[{name}] Error checking model status: {e}")

        print(f"--- Ensemble Ready: {len(self.models)} models loaded ---")

    def predict(self, text):
        if not self.models:
            return {
                "prediction": "No Models Found",
                "confidence": "0%",
                "breakdown": {},
                "status": "demo",
                "message": "Please train at least one model to get real predictions."
            }

        votes = []
        breakdown = {}

        for name, module in self.models.items():
            try:
                # Assuming modules have load_and_predict(text)
                # Note: For efficiency, modules *should* cache their loaded model internally 
                # or we should load them once here in __init__ if memory permits.
                # Given the user context "do not run any model because it will slow down",
                # we are implementing the logic but assuming the user will manage the resource hit when they actually run it.
                # Just calling load_and_predict might be slow if it reloads every time.
                # But refactoring all scripts to cache globally is the pattern we used in bert.py/naive_bayes.py (implicit).
                pred = module.load_and_predict(text)
                if pred:
                    votes.append(pred)
                    breakdown[name] = pred
                else:
                    breakdown[name] = "Error/Unknown"
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                breakdown[name] = "Error"

        if not votes:
            return {
                "prediction": "Error",
                "confidence": "0%",
                "breakdown": breakdown,
                "status": "error"
            }

        # Majority Vote
        vote_counts = Counter(votes)
        top_prediction, count = vote_counts.most_common(1)[0]
        confidence = (count / len(votes)) * 100

        return {
            "prediction": top_prediction,
            "confidence": f"{confidence:.1f}% ({count}/{len(votes)} votes)",
            "breakdown": breakdown,
            "status": "success"
        }

# Initialize Ensemble
ensemble = EnsemblePredictor()

@app.route('/')
def home():
    # Pass loaded model names to template
    loaded_models = list(ensemble.models.keys())
    return render_template('index.html', loaded_models=loaded_models)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['review_text']
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = ensemble.predict(text)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
