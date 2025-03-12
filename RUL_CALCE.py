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
has_mps = torch.backends.mps.is_built()
# device = "cpu"
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
# Sequence Data Preparation
SEQUENCE_SIZE = 32



fn = "./datasets/CALCE/CALCE.csv"
df = pd.read_csv(fn)

names = ["capacity_CS2_35", "capacity_CS2_36", "capacity_CS2_37", "capacity_CS2_38"]

scaler = StandardScaler()

for validation_cell in names:
    log_dict = {}
    log_dict["dataset"] = "CALCE"
    log_dict["test"] = validation_cell
    log_dict["train"] = "all other cells"
    
    logger.info(f"validation cell: {validation_cell}")
    test_cell = df[validation_cell].to_numpy().reshape(-1, 1)
    test_cell = scaler.fit_transform(test_cell).flatten().tolist()
    x_train_all = torch.tensor([], dtype=torch.float32).view(-1, SEQUENCE_SIZE, 1)
    y_train_all = torch.tensor([], dtype=torch.float32).view(-1, 1)
    for cell in [i for i in names if i != validation_cell]:
        train_cell = df[cell].to_numpy().reshape(-1, 1)
        train_cell = scaler.fit_transform(train_cell).flatten().tolist()
        x_train, y_train = to_sequences(SEQUENCE_SIZE, train_cell)
        x_train_all = torch.cat((x_train_all, x_train), dim=0)
        y_train_all = torch.cat((y_train_all, y_train), dim=0)


    x_test, y_test = to_sequences(SEQUENCE_SIZE, test_cell)

    train_dataset = TensorDataset(x_train_all, y_train_all)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
