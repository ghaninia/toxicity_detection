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
import pandas as pd

class PersianSwearDetector:
    def __init__(self, model_path=None):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                strip_accents='unicode',
                analyzer='char_wb',
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
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = digits.convert_to_en(text)
        text = re.sub(r'[،,؛;:!؟?٫٬.۔_\-]+', ' ', text)
        text = re.sub(r'(.)\1+', r'\1\1', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.lower()
        return text

    def load_swear_words(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.swear_words = set(data['words'])
                print(f"Loaded {len(self.swear_words)} swear words.")
        except Exception as e:
            raise Exception(f"Error loading swear words: {str(e)}")

    def create_training_data(self, normal_texts_path, swear_texts_path):
        texts = []
        labels = []
        try:
            with open(normal_texts_path, 'r', encoding='utf-8') as f:
                normal_texts = f.readlines()
                texts.extend(normal_texts)
                labels.extend([0] * len(normal_texts))
        except Exception as e:
            raise Exception(f"Error reading normal texts: {str(e)}")
        try:
            with open(swear_texts_path, 'r', encoding='utf-8') as f:
                swear_texts = f.readlines()
                texts.extend(swear_texts)
                labels.extend([1] * len(swear_texts))
        except Exception as e:
            raise Exception(f"Error reading swear texts: {str(e)}")
        return texts, labels

    def create_training_data_from_json(self, swear_json_path, normal_json_path):
        texts = []
        labels = []
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
        if not any(l == 1 for l in labels) or not any(l == 0 for l in labels):
            raise ValueError("Training data must contain at least one sample from each class (0 and 1)")
        return texts, labels

    def train(self, texts, labels, test_size=0.2):
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Invalid training data.")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        return train_score, test_score

    def load_dataset(self, dataset_path):
        """Load and preprocess the labeled dataset."""
        with open(dataset_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # Adjusted to handle a flat list of dictionaries
        df = pd.DataFrame(data)
        df['text'] = df['text'].apply(self.preprocess_text)
        return df

    def train_model(self, dataset_path):
        """Train the model using the labeled dataset."""
        df = self.load_dataset(dataset_path)
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print("Model trained successfully.")
        print("Test accuracy:", self.pipeline.score(X_test, y_test))

    def save_model(self, model_path):
        """Save the trained model to a file."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if not exist
        with open(model_path, 'wb') as file:
            pickle.dump(self.pipeline, file)
        print(f"Model saved to {model_path}.")

    def load_model(self, model_path):
        """Load a trained model from a file."""
        with open(model_path, 'rb') as file:
            self.pipeline = pickle.load(file)
        self.is_trained = True
        print(f"Model loaded from {model_path}.")

    def rule_based_check(self, text):
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        for word in words:
            if word in self.swear_words:
                return True
        for swear in self.swear_words:
            if swear in processed_text:
                return True
        return False

    def predict(self, text):
        """Predict the label of the given text."""
        if not self.is_trained:
            raise ValueError("The model is not trained yet.")
        processed_text = self.preprocess_text(text)
        rule_based_result = self.rule_based_check(processed_text)
        rule_based_conf = 0.7 if rule_based_result else 0.0

        ml_prob = 0.0
        ml_prediction = False
        used_ml = False

        if self.is_trained:
            ml_prob = self.pipeline.predict_proba([processed_text])[0][1]
            ml_prediction = ml_prob >= 0.5
            used_ml = True

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
    MODEL_PATH = "models/model.pkl"
    DATASET_PATH = "dataset/dataset.json"
    detector = PersianSwearDetector()
    try:
        if os.path.exists(DATASET_PATH):
            detector.train_model(DATASET_PATH)
            detector.save_model(MODEL_PATH)
        elif os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            detector.load_model(MODEL_PATH)
        else:
            print("No model or training data found.")
            return
        print("\nEnter your text for checking (type 'exit' to quit):")
        while True:
            text = input("\nInput text: ")
            if text.lower() == 'exit':
                print("Exiting...")
                break
            try:
                result = detector.predict(text)
                print("\nResult:")
                print(f"  Text: {result['text']}")
                print(f"  Processed Text: {result['processed_text']}")
                print(f"  Rule-Based Detection: {result['rule_based_detection']}")
                print(f"  ML Detection: {result['ml_detection']}")
                print(f"  ML Confidence: {result['ml_confidence']:.2f}")
                print(f"  Final Prediction: {result['final_prediction']}")
                print(f"  Confidence: {result['confidence']:.2f}")
            except Exception as e:
                print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
