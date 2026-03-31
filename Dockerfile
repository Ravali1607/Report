# Use Python base image
FROM python:3.10-slim

# Install system dependencies + tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tel \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start app
CMD ["python", "app.py"]