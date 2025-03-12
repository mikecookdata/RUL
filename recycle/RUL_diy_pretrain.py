import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.express as px
from logger import logger
has_mps = torch.backends.mps.is_built()
# device = "cpu"
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"++++++  Using device: {device}   ++++++")
logger.info("without init weights")
logger.info("train on 36, 37, 38, test on 35")


fn = "./datasets/CALCE/CALCE.csv"
df = pd.read_csv(fn)


spots_train = df["capacity_CS2_36"].to_numpy().reshape(-1, 1)
spots_test = df["capacity_CS2_35"].to_numpy().reshape(-1, 1)

scaler = StandardScaler()
spots_train = scaler.fit_transform(spots_train).flatten().tolist()
spots_test = scaler.transform(spots_test).flatten().tolist()

# Sequence Data Preparation
SEQUENCE_SIZE = 32

def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

x_train, y_train = to_sequences(SEQUENCE_SIZE, spots_train)
x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)

# Setup data loaders for batch
'''TensorDataset
- TensorDataset is a special PyTorch wrapper that combines muliple tensors into a single dataset
- x_train and y_train must have the same first dimension
- this allows easy retrieval of (input, target) pairs during training
- makes integration with DataLoader seamless
'''

'''DataLoader
- Batches data for efficienct training
- Shuffles data befoe each epoch to prevent model from memorizing the order
It randomly select batch(32) size from train_dataset and group them into a batch [32, (sequence)10, 1] for x_train and [32, 1] for y_train
return as PyTorch Tensor for training
'''

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
    #     self.apply(self._init_weights) # what's the weights if not initialized?
    #     # see link https://chatgpt.com/c/67ce20e9-7774-8009-85d9-dcce0feca042
        
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

model = TransformerModel().to(device)


# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 1000
early_stop_count = 0
min_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        train_losses.append(loss.item())
        loss.backward() # calculate gradients
        optimizer.step() # update weights based on gradients and learning rate

    train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        logger.info("Early stopping!")
        break
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
# Evaluation
from sklearn.metrics import mean_absolute_error
# Evaluation
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(y_batch.squeeze().tolist())
        
# Convert to numpy arrays and inverse transform
predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

predictions_inv = scaler.inverse_transform(predictions)
actuals_inv = scaler.inverse_transform(actuals)

# Compute Scores
rmse = np.sqrt(np.mean((predictions_inv - actuals_inv) ** 2))
mae = mean_absolute_error(actuals_inv, predictions_inv)
re = np.mean(np.abs((actuals_inv - predictions_inv) / actuals_inv))  # Percentage form

logger.info(f"SEQUENCE_SIZE: {SEQUENCE_SIZE}")

logger.info(f"Score (Relative Error): {re:.4f}")
logger.info(f"Score (MAE): {mae:.4f}")
logger.info(f"Score (RMSE): {rmse:.4f}")