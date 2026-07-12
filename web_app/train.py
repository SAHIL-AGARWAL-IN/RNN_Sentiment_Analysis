import os
import re
import pickle
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data programmatically
print("Checking NLTK requirements...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# 1. Text Preprocessing Pipeline
def clean_text(text, stop_words, ps):
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuations and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize and remove stopwords and stem
    tokens = word_tokenize(text)
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

# 2. PyTorch Dataset to load sparse data efficiently without converting the whole matrix to dense in RAM
class SparseDataset(Dataset):
    def __init__(self, X_sparse, y_labels):
        self.X_sparse = X_sparse.tocsr()
        self.y_labels = y_labels.values if hasattr(y_labels, 'values') else y_labels

    def __len__(self):
        return self.X_sparse.shape[0]

    def __getitem__(self, idx):
        x_dense = self.X_sparse[idx].toarray().squeeze()
        y_val = self.y_labels[idx]
        return torch.tensor(x_dense, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# 3. RNN Model definition
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
        # Slicing the output of the last sequence step (seq_len is 1 for our TF-IDF feed)
        out = self.fc(out[:, -1:])
        return out

def main():
    start_time = time.time()
    
    # Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(root_dir, "IMDB Dataset.csv")
    assets_dir = os.path.join(root_dir, "web_app", "model_assets")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"IMDB Dataset.csv not found at {dataset_path}. Please place it in the workspace root.")
        
    df = pd.read_csv(dataset_path)
    
    print("Removing duplicates...")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Preprocessing review texts...")
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    
    # Process review texts
    t0 = time.time()
    df["clean_review"] = df["review"].apply(lambda x: clean_text(x, stop_words, ps))
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    print("Encoding labels...")
    le = LabelEncoder()
    df["sentiment_code"] = le.fit_transform(df["sentiment"]) # positive=1, negative=0
    
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_review"])
    y = df["sentiment_code"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create PyTorch datasets
    train_dataset = SparseDataset(X_train, y_train)
    test_dataset = SparseDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    input_size = 5000
    model = RNN(input_size=input_size, hidden_size=128, num_layers=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training the RNN model...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Add sequence length dimension: shape (batch_size, 1, input_size)
            xb = xb.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(xb)
            
            # Apply sigmoid to outputs and squeeze to match yb dimensions
            outputs = torch.sigmoid(outputs.squeeze())
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * xb.size(0)
            
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        
    # Evaluate model
    print("Evaluating model...")
    model.eval()
    correct_vals = 0
    tot_vals = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(1)
            outputs = model(xb)
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            tot_vals += yb.size(0)
            correct_vals += (predicted == yb).sum().item()
            
    accuracy = (correct_vals / tot_vals) * 100
    print(f"Training finished. Test Accuracy: {accuracy:.2f}%")
    
    # Save Model Assets
    print("Saving model assets...")
    model_path = os.path.join(assets_dir, "model.pth")
    tfidf_path = os.path.join(assets_dir, "tfidf.pkl")
    config_path = os.path.join(assets_dir, "config.json")
    
    # Save PyTorch state dict
    torch.save(model.state_dict(), model_path)
    
    # Save TF-IDF Vectorizer
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
        
    # Save configuration and stats
    config = {
        "input_size": input_size,
        "hidden_size": 128,
        "num_layers": 1,
        "classes": list(le.classes_), # ['negative', 'positive']
        "accuracy": accuracy,
        "training_time_seconds": time.time() - start_time,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Assets saved successfully to {assets_dir}!")
    print(f"Total script run time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
