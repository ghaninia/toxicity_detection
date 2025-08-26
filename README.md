# 🛡️ Persian Swear Detector

A robust and easy-to-use Python tool for detecting Persian (Farsi) offensive text using both rule-based and machine learning (ML) approaches.

## 🚀 Features

- **Hybrid Detection**: Combines rule-based and ML-based detection for high accuracy
- **Confidence Scores**: Provides confidence levels for predictions
- **Persian Language Support**: Handles Persian text preprocessing and normalization
- **CLI Interface**: Simple command-line interface for quick testing
- **Model Persistence**: Save and load trained models for fast deployment

## 📦 Project Structure

```
├── swear_detector.py         # Main detector script
├── requirements.txt          # Python dependencies
├── dataset/
│   └── dataset.json         # Labeled dataset for training (Offensive/Normal)
├── models/
│   └── model.pkl            # Trained ML model
├── Dockerfile               # Docker support
├── docker-compose.yml       # Docker Compose config
└── README.md               # Documentation
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

Run the detector:
```bash
python3 swear_detector.py
```


Each prediction includes:
- Original text
- Final prediction (Offensive/Normal)
- Confidence score
- ML confidence score (for offensive predictions)

## 📊 Dataset

The project uses a labeled dataset (`dataset.json`) containing:
- Offensive texts: Inappropriate or offensive content
- Normal texts: Regular, non-offensive content

## 🤖 Model

The system uses a hybrid approach:
1. Machine Learning: TF-IDF + Logistic Regression
2. Rule-based detection
3. Combined scoring for final prediction

## 🐳 Docker Support

Build and run with Docker:
```bash
docker-compose up --build
```

