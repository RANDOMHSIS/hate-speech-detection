# hate_speech_inference.py
# Standalone inference module — no training/augmentation dependencies needed

import torch
import torch.nn as nn
import json
import re

# ============================================================
# TOKENIZER (must match training exactly)
# ============================================================
def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def encode(text, vocab, max_len):
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab['<UNK>']) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ============================================================
# MODEL ARCHITECTURE (must match training exactly)
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(final_hidden)
        return self.fc(out)

# ============================================================
# LOADER — call this once when your bot starts up
# ============================================================
class HateSpeechDetector:
    def __init__(self, model_path, vocab_path, config_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        with open(config_path, 'r') as f:
            config = json.load(f)
        self.max_len = config['max_len']

        self.model = LSTMClassifier(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, text):
        """Returns (label: str, confidence: float)"""
        ids = encode(text, self.vocab, self.max_len)
        tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred].item()
        label = "HATE" if pred == 1 else "NON-HATE"
        return label, confidence

# ============================================================
# USAGE IN YOUR DISCORD BOT:
# ============================================================
# detector = HateSpeechDetector(
#     model_path='best_lstm_model.pt',
#     vocab_path='vocab.json',
#     config_path='model_config.json'
# )
#
# label, confidence = detector.predict(message.content)
# if label == "HATE" and confidence > 0.7:
#     # flag/moderate the message