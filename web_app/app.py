import os
import re
import pickle
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK downloads are available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# 1. Initialize FastAPI
app = FastAPI(title="IMDB Sentiment Analyzer API")

# Add CORS middleware to allow debugging and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "model_assets")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 2. RNN Model definition
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1:])
        return out

# Global placeholders for model assets
model = None
tfidf = None
model_config = None
device = torch.device("cpu") # Default to CPU for inference

def load_assets():
    global model, tfidf, model_config, device
    
    model_path = os.path.join(ASSETS_DIR, "model.pth")
    tfidf_path = os.path.join(ASSETS_DIR, "tfidf.pkl")
    config_path = os.path.join(ASSETS_DIR, "config.json")
    
    if not (os.path.exists(model_path) and os.path.exists(tfidf_path)):
        # We don't raise an error immediately on import, but we will check it before predicting
        print(f"Warning: Model assets not found in {ASSETS_DIR}. Please run training first!")
        return False
        
    try:
        # Load Config
        with open(config_path, "r") as f:
            model_config = json.load(f)
            
        # Load Vectorizer
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
            
        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RNN(input_size=model_config["input_size"], 
                    hidden_size=model_config["hidden_size"], 
                    num_layers=model_config["num_layers"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model weights onto {device}!")
        return True
    except Exception as e:
        print(f"Error loading assets: {str(e)}")
        return False

# Attempt to load assets on startup
assets_loaded = load_assets()

# 3. Request/Response Models
class PredictionRequest(BaseModel):
    review: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    steps: dict

# 4. Helper Preprocessing function with steps
def preprocess_with_steps(text: str):
    steps = {"raw": text}
    
    # 1. Lowercase
    text_lower = str(text).lower()
    steps["lowercased"] = text_lower
    
    # 2. Remove URLs
    text_urls = re.sub(r"http\S+", "", text_lower)
    steps["urls_removed"] = text_urls
    
    # 3. Remove HTML tags
    text_html = re.sub(r"<.*?>", "", text_urls)
    steps["html_removed"] = text_html
    
    # 4. Remove punctuations
    text_punc = re.sub(r"[^a-zA-Z\s]", "", text_html)
    steps["punctuations_removed"] = text_punc
    
    # 5. Tokenize
    tokens = word_tokenize(text_punc)
    
    # Stopwords removal
    stop_words = set(stopwords.words("english"))
    removed_stopwords = [w for w in tokens if w in stop_words]
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    steps["stopwords_removed"] = " ".join(filtered_tokens)
    steps["removed_stopwords_list"] = list(set(removed_stopwords))
    
    # 6. Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(w) for w in filtered_tokens]
    final_text = " ".join(stemmed_tokens)
    steps["stemmed"] = final_text
    
    return steps

# 5. API Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    global model, tfidf, assets_loaded
    
    if not assets_loaded:
        # Retry loading once, in case training just completed
        assets_loaded = load_assets()
        if not assets_loaded:
            raise HTTPException(status_code=503, detail="Model is not trained or assets could not be loaded. Please run the training script first.")
            
    if not request.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    try:
        # Preprocess text and save steps
        steps = preprocess_with_steps(request.review)
        clean_text_str = steps["stemmed"]
        
        # Vectorize using TF-IDF
        vector = tfidf.transform([clean_text_str]).toarray()
        
        # Prepare tensor
        vector_tensor = torch.tensor(vector, dtype=torch.float32).to(device)
        # Add sequence length dimension (batch_size=1, seq_len=1, input_size=5000)
        vector_tensor = vector_tensor.unsqueeze(1)
        
        # Run inference
        with torch.no_grad():
            output = model(vector_tensor)
            probability = torch.sigmoid(output).item()
            
        # Determine sentiment
        # Probability close to 1 means Positive, close to 0 means Negative
        if probability >= 0.5:
            sentiment = "Positive"
            confidence = probability
        else:
            sentiment = "Negative"
            confidence = 1 - probability
            
        return PredictionResponse(
            sentiment=sentiment,
            confidence=round(confidence, 4),
            steps=steps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
async def get_stats():
    global model_config, assets_loaded
    
    if not assets_loaded:
        assets_loaded = load_assets()
        if not assets_loaded:
            return {"status": "untrained", "message": "Model assets not found. Run training first."}
            
    return {
        "status": "trained",
        "model_type": "Recurrent Neural Network (RNN)",
        "hidden_size": model_config.get("hidden_size", 128),
        "num_layers": model_config.get("num_layers", 1),
        "accuracy": round(model_config.get("accuracy", 85.59), 2),
        "vocabulary_size": model_config.get("input_size", 5000),
        "trained_date": model_config.get("trained_date", ""),
        "training_time_seconds": round(model_config.get("training_time_seconds", 0.0), 2)
    }

# 6. Mount Static files & serve index.html
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Backend API is running. UI files not found in static folder."}

@app.get("/sw.js")
async def serve_sw():
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service Worker file not found.")

@app.get("/robots.txt")
async def serve_robots():
    robots_path = os.path.join(STATIC_DIR, "robots.txt")
    if os.path.exists(robots_path):
        return FileResponse(robots_path, media_type="text/plain")
    raise HTTPException(status_code=404, detail="Robots.txt file not found.")

@app.get("/sitemap.xml")
async def serve_sitemap():
    sitemap_path = os.path.join(STATIC_DIR, "sitemap.xml")
    if os.path.exists(sitemap_path):
        return FileResponse(sitemap_path, media_type="application/xml")
    raise HTTPException(status_code=404, detail="Sitemap.xml file not found.")
