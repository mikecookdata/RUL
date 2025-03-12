import json
from datetime import datetime
import torch

def append_to_json_file(file_path, new_data):
    new_data["datetime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Try to read existing data
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):  # Ensure it's a list
                data = []
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, start a new list
        data = []

    # Append new dictionary
    data.append(new_data)

    # Write updated list back to file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
        
def to_sequences(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i:(i + seq_size)]
        after_window = obs[i + seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)