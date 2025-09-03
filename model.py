import torch
import torch.nn as nn
import numpy as np
from config import SEQ_LEN, DEVICE, H

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
class LSTMDirect(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, horizon=H):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # on prend le dernier pas de la séquence
        return out

def predict_future(model, last_seq, n_steps, scaler_y, H=5, device=DEVICE):
    """
    Predict n_steps into the future using a multi-horizon LSTM.
    
    last_seq : np.array de forme (SEQ_LEN, n_features) -> dernière séquence connue
    n_steps  : nombre total de pas à prédire
    scaler_y : scaler pour inverser la cible
    H        : horizon du modèle (nombre de pas prédits à la fois)
    """
    model.eval()
    seq = last_seq.copy()
    predictions = []

    with torch.no_grad():
        steps = n_steps
        while steps > 0:
            input_seq = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_seq).cpu().numpy().ravel()  # shape (H,)
            
            # ne garder que ce qui reste à prédire
            to_take = min(steps, H)
            predictions.extend(pred[:to_take])
            
            # réinjecter dans la séquence
            next_feat = np.zeros(seq.shape[1])
            next_feat[0] = pred[0]  # mise à jour uniquement du prix (ou target)
            seq = np.vstack([seq, next_feat])
            
            steps -= to_take

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler_y.inverse_transform(predictions)
