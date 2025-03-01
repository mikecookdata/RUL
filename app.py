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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load data
Battery_list = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
Battery = np.load("datasets/CALCE/CALCE.npy", allow_pickle=True)
Battery = Battery.item()

print(Battery.keys())
print(Battery["CS2_35"])


# util functions
def drop_outlier(array, count, bins):
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i : i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    # text:list of capacity
    x, y = [], []
    for i in range(len(text) - window_size):
        sequence = text[i : i + window_size]
        target = text[i + window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name]["capacity"]
    train_data, test_data = (
        data_sequence[: window_size + 1],
        data_sequence[window_size + 1 :],
    )
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(
                text=v["capacity"], window_size=window_size
            )
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return (
        abs(true_re - pred_re) / true_re if abs(true_re - pred_re) / true_re <= 1 else 1
    )


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
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
        """
        Args:
            input_size: the feature size of input data (required).
            hidden_dim: the hidden size of AutoEncoder (required).
            noise_level: the noise level added in Autoencoder (optional).
        """
        super(Autoencoder, self).__init__()
        self.noise_level = noise_level
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_size)

    def encoder(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        return h1

    def mask(self, x):
        corrupted_x = x + self.noise_level * torch.randn_like(x)
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
    def __init__(self, feature_len, feature_size, dropout=0.0):
        """
        Args:
            feature_len: the feature length of input data (required).
            feature_size: the feature size of input data (required).
            dropout: the dropout rate (optional).
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, feature_size, 2).float()
            * (-math.log(10000.0) / feature_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe

        return x


class Net(nn.Module):
    def __init__(
        self,
        feature_size=16,
        hidden_dim=32,
        feature_num=1,
        num_layers=1,
        nhead=1,
        dropout=0.0,
        noise_level=0.01,
    ):
        """
        Args:
            feature_size: the feature size of input data (required).
            hidden_dim: the hidden size of Transformer block (required).
            feature_num: the number of features, such as capacity, voltage, and current; set 1 for only sigle feature (optional).
            num_layers: the number of layers of Transformer block (optional).
            nhead: the number of heads of multi-attention in Transformer block (optional).
            dropout: the dropout rate of Transformer block (optional).
            noise_level: the noise level added in Autoencoder (optional).
        """
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size / 2)
        input_size = self.auto_hidden

        if feature_num == 1:
            # Transformer treated as an Encoder when modeling for a sigle feature like only capacity data
            self.pos = PositionalEncoding(
                feature_len=feature_num, feature_size=input_size
            )
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
        elif feature_num > 1:
            # Transformer treated as a sequence model when modeling for multi-features like capacity, voltage, and current data
            self.pos = PositionalEncoding(
                feature_len=input_size, feature_size=feature_num
            )
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=feature_num,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.linear = nn.Linear(feature_num * self.auto_hidden, 1)
        self.autoencoder = Autoencoder(
            input_size=feature_size,
            hidden_dim=self.auto_hidden,
            noise_level=noise_level,
        )

    def forward(self, x):
        batch_size, feature_num, feature_size = x.shape
        out, decode = self.autoencoder(x)
        if feature_num > 1:
            out = out.reshape(batch_size, -1, feature_num)
        out = self.pos(out)
        out = self.cell(
            out
        )  # sigle feature: (batch_size, feature_num, auto_hidden) or multi-features: (batch_size, auto_hidden, feature_num)
        out = out.reshape(batch_size, -1)  # (batch_size, feature_num*auto_hidden)
        out = self.linear(out)  # out shape: (batch_size, 1)

        return out, decode


def train(
    lr=0.01,
    feature_size=8,
    feature_num=1,
    hidden_dim=32,
    num_layers=1,
    nhead=1,
    dropout=0.0,
    epochs=1000,
    weight_decay=0.0,
    seed=0,
    alpha=0.0,
    noise_level=0.0,
    metric="re",
    device="cpu",
):
    """
    Args:
        lr: learning rate for training (required).
        feature_size: the feature size of input data (required).
        feature_num: the number of features, such as capacity, voltage, and current; set 1 for only sigle feature (optional).
        hidden_dim: the hidden size of Transformer block (required).
        num_layers: the number of layers of Transformer block (optional).
        nhead: the number of heads of multi-attention in Transformer block (optional).
        dropout: the dropout rate of Transformer block (optional).
        epochs:
        weight_decay:
        seed: (optional).
        alpha: (optional).
        noise_level: the noise level added in Autoencoder (optional).
        metric: (optional).
        device: (optional).
    """
    score_list, fixed_result_list, moving_result_list = [], [], []
    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            Battery, name, feature_size
        )
        test_sequence = train_data + test_data
        # print('sample size: {}'.format(len(train_x)))

        model = Net(
            feature_size=feature_size,
            hidden_dim=hidden_dim,
            feature_num=K,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            noise_level=noise_level,
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_fixed_slice, y_moving_slice = [0], [], []
        rmse, re = 1, 1
        score_, score = [1], [1]
        for epoch in range(epochs):
            x = np.reshape(
                train_x / Rated_Capacity, (-1, feature_num, feature_size)
            ).astype(np.float32)
            y = np.reshape(train_y / Rated_Capacity, (-1, 1)).astype(np.float32)

            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            x = x.repeat(1, K, 1)
            output, decode = model(x)
            output = output.reshape(-1, 1)
            loss = criterion(output, y) + alpha * criterion(decode, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                test_x = train_data.copy()
                fixed_point_list, moving_point_list = [], []
                t = 0
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(
                        np.array(test_x[-feature_size:]) / Rated_Capacity,
                        (-1, feature_num, feature_size),
                    ).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    x = x.repeat(1, K, 1)
                    pred, _ = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    test_x.append(
                        next_point
                    )  # The test values are added to the original sequence to continue to predict the next point
                    fixed_point_list.append(
                        next_point
                    )  # Saves the predicted value of the last point in the output sequence

                    x = np.reshape(
                        np.array(test_sequence[t : t + feature_size]) / Rated_Capacity,
                        (-1, 1, feature_size),
                    ).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    x = x.repeat(1, K, 1)
                    pred, _ = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    moving_point_list.append(
                        next_point
                    )  # Saves the predicted value of the last point in the output sequence
                    t += 1

                y_fixed_slice.append(fixed_point_list)  # Save all the predicted values
                y_moving_slice.append(moving_point_list)

                loss_list.append(loss)
                rmse = evaluation(y_test=test_data, y_predict=y_fixed_slice[-1])
                re = relative_error(
                    y_test=test_data,
                    y_predict=y_fixed_slice[-1],
                    threshold=Rated_Capacity * 0.7,
                )
                # print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))

            if metric == "re":
                score = [re]
            elif metric == "rmse":
                score = [rmse]
            else:
                score = [re, rmse]
            if (loss < 1e-3) and (score_[0] < score[0]):
                break
            score_ = score.copy()

        score_list.append(score_)
        fixed_result_list.append(train_data.copy() + y_fixed_slice[-1])
        moving_result_list.append(train_data.copy() + y_moving_slice[-1])

    return score_list, fixed_result_list, moving_result_list


# Setting and training for overall performance

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
metric = "re"
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K = 16

SCORE = []
for seed in tqdm(range(5)):
    print("seed:{}".format(seed))
    score_list, _, _ = train(
        lr=lr,
        feature_size=feature_size,
        feature_num=feature_num,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=nhead,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
        dropout=dropout,
        alpha=alpha,
        noise_level=noise_level,
        metric=metric,
        device=device,
    )
    print(np.array(score_list))
    print(metric + "for this seed: {:<6.4f}".format(np.mean(np.array(score_list))))
    for s in score_list:
        SCORE.append(s)
    print("------------------------------------------------------------------")
print(metric + " mean: {:<6.4f}".format(np.mean(np.array(SCORE))))
