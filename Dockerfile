# Use official Python image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY swear_detector.py ./
COPY dataset ./dataset
COPY models ./models

# Set default command
CMD ["python", "swear_detector.py"]

