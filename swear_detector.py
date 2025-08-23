import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from persian_tools import digits
import re
import pickle

class PersianSwearDetector:
    def __init__(self, model_path=None):
        # Create a pipeline for text processing and model
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                ngram_range=(1, 3),  # Use unigrams to trigrams
                max_features=5000,
                strip_accents='unicode',
                analyzer='char_wb',  # Analyze characters with word boundaries
                preprocessor=self.preprocess_text
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        self.swear_words = set()
        self.is_trained = False
        # Load model if exists
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_text(self, text):
        """
        Advanced text preprocessing for Persian text
        """
        if not isinstance(text, str):
            return ""
        # Convert Persian digits to English
        text = digits.convert_to_en(text)
        # Remove special characters
        text = re.sub(r'[،,؛;:!؟?٫٬.۔_\-]+', ' ', text)
        # Replace repeated characters
        text = re.sub(r'(.)\1+', r'\1\1', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase
        text = text.lower()
        return text

    def load_swear_words(self, json_path):
        """
        Load swear words from a JSON file
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.swear_words = set(data['words'])
                print(f"Loaded {len(self.swear_words)} swear words.")
        except Exception as e:
            raise Exception(f"Error loading swear words: {str(e)}")

    def create_training_data(self, normal_texts_path, swear_texts_path):
        """
        Create training dataset from text files
        """
        texts = []
        labels = []
        # Read normal texts
        try:
            with open(normal_texts_path, 'r', encoding='utf-8') as f:
                normal_texts = f.readlines()
                texts.extend(normal_texts)
                labels.extend([0] * len(normal_texts))
        except Exception as e:
            raise Exception(f"Error reading normal texts: {str(e)}")
        # Read swear texts
        try:
            with open(swear_texts_path, 'r', encoding='utf-8') as f:
                swear_texts = f.readlines()
                texts.extend(swear_texts)
                labels.extend([1] * len(swear_texts))
        except Exception as e:
            raise Exception(f"Error reading swear texts: {str(e)}")
        return texts, labels

    def create_training_data_from_json(self, swear_json_path, normal_json_path):
        """
        Create training data from swear words JSON and normal sentences JSON file
        """
        texts = []
        labels = []
        # Load swear words
        try:
            with open(swear_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                swear_texts = data.get('words', [])
                if not swear_texts:
                    raise ValueError(f"No swear words found in {swear_json_path}")
                texts.extend(swear_texts)
                labels.extend([1] * len(swear_texts))
        except Exception as e:
            raise Exception(f"Error reading swear words: {str(e)}")
        # Load normal sentences from JSON file
        try:
            with open(normal_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                normal_texts = data.get('words', [])
                if not normal_texts:
                    raise ValueError(f"No normal words found in {normal_json_path}")
                texts.extend(normal_texts)
                labels.extend([0] * len(normal_texts))
        except Exception as e:
            raise Exception(f"Error reading normal words: {str(e)}")
        # Check that both classes are present
        if not any(l == 1 for l in labels) or not any(l == 0 for l in labels):
            raise ValueError("Training data must contain at least one sample from each class (0 and 1)")
        return texts, labels

    def train(self, texts, labels, test_size=0.2):
        """
        Train the model with input data
        """
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Invalid training data.")
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        # Train model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        # Evaluate model
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        return train_score, test_score

    def save_model(self, model_path):
        """
        Save the trained model to a file
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

    def load_model(self, model_path):
        """
        Load the model from a file
        """
        try:
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def rule_based_check(self, text):
        """
        Rule-based check for swear words in text
        """
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        # Check for exact words
        for word in words:
            if word in self.swear_words:
                return True
        # Check for substrings
        for swear in self.swear_words:
            if swear in processed_text:
                return True
        return False

    def predict(self, text):
        """
        Predict using ML model and rule-based method with weighted combination
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid input text.")

        processed_text = self.preprocess_text(text)
        rule_based_result = self.rule_based_check(processed_text)
        rule_based_conf = 0.7 if rule_based_result else 0.0  # default confidence for rule-based

        ml_prob = 0.0
        ml_prediction = False
        used_ml = False

        # Use ML prediction if model is trained
        if self.is_trained:
            ml_prob = self.pipeline.predict_proba([processed_text])[0][1]
            ml_prediction = ml_prob >= 0.5
            used_ml = True

        # Combine rule-based and ML using weighted sum
        # weight_rule = 0.3, weight_ml = 0.7
        final_confidence = (0.3 * rule_based_conf) + (0.7 * ml_prob)
        final_prediction = final_confidence >= 0.5

        return {
            "text": text,
            "processed_text": processed_text,
            "rule_based_detection": bool(rule_based_result),
            "ml_detection": bool(ml_prediction),
            "ml_confidence": float(ml_prob),
            "final_prediction": bool(final_prediction),
            "confidence": float(final_confidence),
            "used_ml_prediction": bool(used_ml)
        }

def main():
    # File paths
    MODEL_PATH = "models/swear_detector_model.pkl"
    SWEAR_WORDS_PATH = "dataset/swear_words.json"
    NORMAL_WORDS_PATH = "dataset/normal_words.json"
    detector = PersianSwearDetector()
    try:
        # Train model with dataset if available
        if os.path.exists(SWEAR_WORDS_PATH) and os.path.exists(NORMAL_WORDS_PATH):
            texts, labels = detector.create_training_data_from_json(SWEAR_WORDS_PATH, NORMAL_WORDS_PATH)
            detector.train(texts, labels)
            detector.save_model(MODEL_PATH)
        elif os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            detector.load_model(MODEL_PATH)
        else:
            print(json.dumps({"status": "error", "message": "No model or training data found."}))
            return
        # Main input loop
        print("\nEnter your text for checking (type 'exit' to quit):")
        while True:
            text = input("\nInput text: ")
            if text.lower() == 'exit':
                print(json.dumps({"status": "ok", "message": "Exiting..."}))
                break
            try:
                result = detector.predict(text)
                print(json.dumps({"status": "ok", "result": result}, ensure_ascii=False, indent=2))
            except Exception as e:
                print(json.dumps({"status": "error", "message": str(e)}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    main()
