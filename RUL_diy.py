import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import plotly.express as px
from logger import logger
from model import TransformerModel
from utils import append_to_json_file, to_sequences
import json
has_mps = torch.backends.mps.is_built()
# device = "cpu"
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

log_dict = {}

logger.info(f"++++++  Using device: {device}   ++++++")
log_dict["device"] = device



fn = "./datasets/CALCE/CALCE.csv"
df = pd.read_csv(fn)

log_dict["dataset"] = "CALCE"
log_dict["test"] = "CS2_35"
log_dict["train"] = "CS2_36, CS2_37, CS2_38"



spots_train_1 = df["capacity_CS2_36"].to_numpy().reshape(-1, 1)
spots_train_2 = df["capacity_CS2_37"].to_numpy().reshape(-1, 1)
spots_train_3 = df["capacity_CS2_38"].to_numpy().reshape(-1, 1)

spots_test = df["capacity_CS2_35"].to_numpy().reshape(-1, 1)

scaler = StandardScaler()
spots_train_1 = scaler.fit_transform(spots_train_1).flatten().tolist()
spots_train_2 = scaler.fit_transform(spots_train_2).flatten().tolist()
spots_train_3 = scaler.fit_transform(spots_train_3).flatten().tolist()

spots_test = scaler.transform(spots_test).flatten().tolist()

# Sequence Data Preparation
SEQUENCE_SIZE = 32

x_train_1, y_train_1 = to_sequences(SEQUENCE_SIZE, spots_train_1)
x_train_2, y_train_2 = to_sequences(SEQUENCE_SIZE, spots_train_2)
x_train_3, y_train_3 = to_sequences(SEQUENCE_SIZE, spots_train_3)
x_train = torch.cat((x_train_1, x_train_2, x_train_3), dim=0)
y_train = torch.cat((y_train_1, y_train_2, y_train_3), dim=0)
x_test, y_test = to_sequences(SEQUENCE_SIZE, spots_test)

# Setup data loaders for batch
'''TensorDataset
- TensorDataset is a special PyTorch wrapper that combines muliple tensors into a single dataset
- x_train and y_train must have the same first dimension
- this allows easy retrieval of (input, target) pairs during training
- makes integration with DataLoader seamless


DataLoader
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
log_dict["SEQUENCE_SIZE"] = SEQUENCE_SIZE
logger.info(f"Score (Relative Error): {re:.4f}")
log_dict["Score (Relative Error)"] = re
logger.info(f"Score (MAE): {mae:.4f}")
log_dict["Score (MAE)"] = mae
logger.info(f"Score (RMSE): {rmse:.4f}")
log_dict["Score (RMSE)"] = rmse

# save log_dict
append_to_json_file('log_dict.json', log_dict)