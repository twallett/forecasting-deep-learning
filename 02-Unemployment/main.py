#%%
from model import Forecast
from utils import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Hyperparameters ---------------------------------------------------------------------

SEED = 123

# 'MLP', 'RNN', 'LSTM', 'GRU'
# Note: MLP and RNN require WINDOW as INPUT_DIM

MODEL = 'GRU'

ASPLIT = 0.8

WINDOW = 7

BATCH_SIZE = 8

INPUT_DIM = 1
HIDDEN_DIM = 16
OUTPUT_DIM = 1
LAYERS = 1
LR = 1e-02
EPOCHS = 500

PATIENCE = 150

# Reproducibility ---------------------------------------------------------------------

seed_everything(SEED)

# Load Data ---------------------------------------------------------------------------
data = pd.read_csv('FRED.csv',
                   index_col=0)

# Scale Data --------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Unemployment'].values.reshape(-1, 1))

# Split Data --------------------------------------------------------------------------
split = int(len(scaled_data) * ASPLIT)
train = scaled_data[:split]
test = scaled_data[split:]

# PyTorch Dataset ---------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window=1):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window - 1

    def __getitem__(self, idx):
        X = self.data[idx:(idx + self.window), 0]
        Y = self.data[idx + self.window, 0]
        return torch.tensor(X).float(), torch.tensor(Y).float()
    
train_dataset = TimeSeriesDataset(train,
                                  window = WINDOW)
test_dataset = TimeSeriesDataset(test,
                                 window = WINDOW)

# PyTorch Dataloader ------------------------------------------------------------------
train_loader = DataLoader(train_dataset,
                          batch_size = BATCH_SIZE)
test_loader = DataLoader(test_dataset,
                         batch_size = BATCH_SIZE)

# PyTorch Model -----------------------------------------------------------------------
model = Forecast(model = MODEL,
                    input_dim = INPUT_DIM, 
                    hidden_dim = HIDDEN_DIM, 
                    output_dim = OUTPUT_DIM, 
                    num_layers = LAYERS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

early_stopping = EarlyStopping(model = MODEL, patience = PATIENCE)

# Train-Test Loop ----------------------------------------------------------------------
results = []
for epoch in range(EPOCHS):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            
    model.eval()
    total_loss = 0
    predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:        
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss
            
            predictions.append(outputs.numpy())
            
        early_stopping(loss, MODEL)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        avg_loss = total_loss / len(test_loader)
        results.append(avg_loss.item())

# Results Aggregation -----------------------------------------------------------------
mean_loss = np.mean(results)
std_loss = np.std(results)
print(f'Mean MSE: {mean_loss:.4f}')
print(f'Standard Deviation of MSE: {std_loss:.4f}')

# Visualization ------------------------------------------------------------------------
predictions = scaler.inverse_transform(np.concatenate(predictions).reshape(-1, 1))

plt.plot(range(0, len(train)), data['Unemployment'][:split], label = 'Train')
plt.plot(range(len(train), len(train) + len(test) - WINDOW - 1), data['Unemployment'][split:-WINDOW-1], label = 'Test')
plt.plot(range(len(train), len(train) + len(test) - WINDOW - 1), predictions, label = 'Predictions')
plt.title(f'{MODEL} Forecast of Unemployment')
plt.xlabel('Time')
plt.ylabel('Unemployment')
plt.legend()
plt.show()

# Metrics -------------------------------------------------------------------------------
print(f"\n mse: {mean_squared_error(data['Unemployment'][split:-WINDOW-1].values,predictions).round(4)}")
print(f"\n mae: {mean_absolute_error(data['Unemployment'][split:-WINDOW-1].values,predictions).round(4)}")

# %%
