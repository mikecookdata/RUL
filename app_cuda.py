import numpy as np
import random
import math
import time
import os
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
Battery_list = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
Battery = np.load("datasets/CALCE/CALCE.npy", allow_pickle=True)
Battery = Battery.item()

# Util functions
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Build model
class Autoencoder(nn.Module):
    def __init__(self, input_size=16, hidden_dim=8, noise_level=0.01):
        super(Autoencoder, self).__init__()
        self.noise_level = noise_level
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_size)

    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1

    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x).to(device)
        return corrupted_x

    def decoder(self, x):
        h2 = self.fc2(x)
        return h2

    def forward(self, x):
        out = self.mask(x)
        encode = self.encoder(out)
        decode = self.decoder(encode)
        return encode, decode

class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(feature_len, feature_size, device=device)
        position = torch.arange(0, feature_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_size, 2, device=device).float() * (-math.log(10000.0) / feature_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x

class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, feature_num=1, num_layers=1, nhead=1, dropout=0.0, noise_level=0.01):
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size / 2)
        self.pos = PositionalEncoding(feature_len=feature_num, feature_size=self.auto_hidden)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.auto_hidden, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True).to(device)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(feature_num * self.auto_hidden, 1).to(device)
        self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level).to(device)

    def forward(self, x):
        x = x.to(device)
        out, decode = self.autoencoder(x)
        out = self.pos(out)
        out = self.cell(out)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out, decode

def train(lr=0.01, feature_size=8, feature_num=1, hidden_dim=32, num_layers=1, nhead=1, dropout=0.0, epochs=1000, weight_decay=0.0, seed=0, alpha=0.0, noise_level=0.0, metric="re", device="cpu"):
    setup_seed(seed)
    model = Net(feature_size=feature_size, hidden_dim=hidden_dim, feature_num=feature_num, num_layers=num_layers, nhead=nhead, dropout=dropout, noise_level=noise_level).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss().to(device)
    
    # Simulated training loop
    for epoch in range(epochs):
        x = torch.randn((32, feature_num, feature_size)).to(device)  # Fake training data
        y = torch.randn((32, 1)).to(device)  # Fake training labels
        optimizer.zero_grad()
        output, decode = model(x)
        loss = criterion(output, y) + alpha * criterion(decode, x)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

# Training Configuration
Rated_Capacity = 1.1
feature_size = 64
feature_num = 1
dropout = 0.0
epochs = 500
nhead = 1
hidden_dim = 16
num_layers = 3
lr = 0.001  
weight_decay = 0.0
noise_level = 0.0
alpha = 0.1
metric = 're'
seed = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K = 16

SCORE = []
for seed in tqdm(range(5)):
    print('seed:{}'.format(seed))
    score_list, _, _ = train(lr=lr, feature_size=feature_size, feature_num=feature_num, hidden_dim=hidden_dim, num_layers=num_layers, 
                             nhead=nhead, weight_decay=weight_decay, epochs=epochs, seed=seed, dropout=dropout, alpha=alpha, 
                             noise_level=noise_level, metric=metric, device=device)
    print(np.array(score_list))
    print(metric + 'for this seed: {:<6.4f}'.format(np.mean(np.array(score_list))))
    for s in score_list:
        SCORE.append(s)
    print('------------------------------------------------------------------')
print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))