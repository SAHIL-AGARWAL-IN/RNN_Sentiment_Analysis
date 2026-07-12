# IMDB Movie Review Sentiment Analyzer (RNN)

This project features a Recurrent Neural Network (RNN) built in PyTorch to classify the sentiment of movie reviews (Positive or Negative). It includes a robust web-based UI for real-time predictions, detailed step-by-step visualization of text preprocessing, and a historical log of queries.

## Features
*   **Deep Learning Model**: PyTorch-based RNN model trained on the IMDB Movie Reviews Dataset.
*   **Web Interface**: Sleek, glassmorphic UI featuring a responsive dark mode dashboard.
*   **Preprocessing Pipeline Inspector**: Watch how raw text changes step-by-step: Lowercasing, URL/HTML removal, Punctuation removal, Stopwords removal, and Porter Stemmer Stemming.
*   **Radial Sentiment Gauge**: Visualizes classification probability dynamically.
*   **Analysis History Log**: Persists previous analysis results using local storage.
*   **Zero-Dataset Web App**: The web app only requires lightweight saved model parameters (`model.pth`, ~2.5 MB) and the TF-IDF vectorizer (`tfidf.pkl`, ~150 KB) to run, keeping server deployment footprints minimal.

---

## Getting Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Make sure `IMDB Dataset.csv` is in the root directory, then run:
```bash
python web_app/train.py
```
This script will clean the dataset, train the PyTorch RNN for 10 epochs, and export all required assets to `web_app/model_assets/`.

### 3. Run the Web Interface
Start the FastAPI server:
```bash
uvicorn web_app.app:app --reload
```
Open your browser and navigate to `http://127.0.0.1:8000` to interact with the UI.

---

## Deployment Guide

This app is optimized for hosting on free tiers like **Render** or **Hugging Face Spaces**.

### Deploying to Render
1. Create a GitHub repository and push your project code (the `.gitignore` automatically excludes the large 66MB dataset, keeping the upload lightweight).
2. Go to [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration details:
   *   **Runtime**: `Python 3`
   *   **Build Command**: `pip install -r requirements.txt`
   *   **Start Command**: `uvicorn web_app.app:app --host 0.0.0.0 --port $PORT`
5. Click deploy!

### Deploying to Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select **Docker** or **Gradio** as the SDK. (If choosing Docker, use a basic python image that runs `uvicorn web_app.app:app --host 0.0.0.0 --port 7860`).
2. If deploying as a standard web application, Docker provides the most flexibility for FastAPI.
