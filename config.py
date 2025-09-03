# Config
import torch

H = 5
TICKER = "BTC-EUR"
START = "2018-01-01"
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_cols = ["Close", "ma_10", "ma_30", "vol_10", "rsi_14"]

