# 🛡️ Persian Swear Detector

A robust and easy-to-use Python tool for detecting Persian (Farsi) swear words in text using both rule-based and machine learning (ML) approaches.

## 🚀 Features

- **Hybrid Detection**: Combines rule-based and ML-based detection for high accuracy.
- **Customizable Dataset**: Easily extend the list of swear and normal words.
- **Persian Language Support**: Handles Persian text preprocessing and normalization.
- **CLI Interface**: Simple command-line interface for quick testing.
- **Model Persistence**: Save and load trained models for fast deployment.

## 📦 Project Structure

```
├── swear_detector.py         # Main detector script
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── swear_words.json      # List of Persian swear words
│   └── normal_words.json     # List of normal (non-swear) words
├── models/
│   └── swear_detector_model.pkl # Trained ML model
├── Dockerfile                # Docker support
├── docker-compose.yml        # Docker Compose config
└── README.md                 # This file
```

## 🏁 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ghaninia/toxicity_detection.git
cd swear-persian
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Datasets

- Edit `dataset/swear_words.json` and `dataset/normal_words.json` to add your own words.

### 4. Train or Use the Model

- **Train:**
  - If both dataset files exist, the model will train automatically on first run.
- **Use Pretrained Model:**
  - If `models/swear_detector_model.pkl` exists, it will be loaded automatically.

### 5. Run the Detector

```bash
python3 swear_detector.py
```

- Enter your Persian text when prompted.
- Type `exit` to quit.

## 🧠 How It Works

- **Rule-Based:** Checks for exact and substring matches from the swear words list.
- **ML-Based:** Uses a Logistic Regression classifier with TF-IDF features for character n-grams.
- **Hybrid Output:** Both methods are used; the result includes whether ML prediction was used.

## 📝 Example Output

```json
{
  "text": "نمونه متن تستی",
  "processed_text": "نمونه متن تستی",
  "rule_based_detection": false,
  "ml_detection": false,
  "ml_confidence": 0.02,
  "final_prediction": false,
  "confidence": 0.02,
  "used_ml_prediction": true
}
```

## 🛠️ Customization

- Add more words to `dataset/swear_words.json` and `dataset/normal_words.json`.
- Retrain the model by deleting the model file or editing the datasets.


