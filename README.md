# Persian Swear Detector

This project uses machine learning and rule-based methods to detect swear words in Persian text. It loads datasets from the `dataset` directory and can be run locally or inside a Docker container.

## Features
- Detects Persian swear words using both rule-based and ML approaches
- Outputs results in JSON format with a status field
- Easily extensible with new datasets

## Project Structure
```
├── dataset/
│   ├── swear_words.json
│   └── normal_words.json
├── models/
│   └── swear_detector_model.pkl
├── swear_detector.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Requirements
- Python 3.9+
- See `requirements.txt` for Python dependencies
- Docker (optional)
- Docker Compose (optional)

## Installation (Local)
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the detector:
   ```bash
   python swear_detector.py
   ```

## Usage
- Enter your text when prompted. Type `exit` to quit.
- Output will be in JSON format, e.g.:
  ```json
  {
    "status": "ok",
    "result": {
      "text": "متن تستی",
      "processed_text": "...",
      "rule_based_detection": false,
      "ml_detection": false,
      "ml_confidence": 0.01,
      "final_prediction": false,
      "confidence": 0.01
    }
  }
  ```

## Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t persian-swear-detector .
   ```
2. Run the container:
   ```bash
   docker run -it --rm -v $(pwd)/dataset:/app/dataset -v $(pwd)/models:/app/models persian-swear-detector
   ```

## Running with Docker Compose
1. Start the service:
   ```bash
   docker-compose up --build
   ```
2. Attach to the container:
   ```bash
   docker attach swear-detector
   ```

## Dataset
- `dataset/swear_words.json`: List of Persian swear words.
- `dataset/normal_words.json`: List of normal Persian sentences.
