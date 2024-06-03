#%%
from model import Forecast
from utils import *
import os
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

N_SPLITS = 10

WINDOW = 7

BATCH_SIZE = 8

INPUT_DIM = 1
HIDDEN_DIM = 16
OUTPUT_DIM = 1
LAYERS = 1
LR = 1e-02
EPOCHS = 500

PATIENCE = 50

# Reproducibility ---------------------------------------------------------------------

seed_everything(SEED)

# Load Data ---------------------------------------------------------------------------
data = pd.read_csv('FRED.csv',
                   index_col=0)

# Scale Data --------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Unemployment'].values.reshape(-1, 1))

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
    
# Time Series Cross Validation --------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS, max_train_size=720, test_size=80)
results = []
best_model = None
best_loss = float('inf')
best_predictions = None

for fold, (train_idx, test_idx) in enumerate(tscv.split(scaled_data)):
    print(f'Fold {fold+1}/{N_SPLITS}')
    
    train_data = scaled_data[train_idx]
    test_data = scaled_data[test_idx]
    
    train_dataset = TimeSeriesDataset(train_data, window=WINDOW)
    test_dataset = TimeSeriesDataset(test_data, window=WINDOW)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Forecast(model = MODEL,
                     input_dim = INPUT_DIM, 
                     hidden_dim = HIDDEN_DIM, 
                     output_dim = OUTPUT_DIM, 
                     num_layers = LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    early_stopping = EarlyStopping(model = MODEL, patience = PATIENCE)

    # Train-Test Loop ----------------------------------------------------------------------
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
    print(f'Fold {fold+1} MSE: {avg_loss.item():.4f}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model = model
        best_train_idx = train_idx
        best_test_idx = test_idx
        best_predictions = np.concatenate(predictions)

# Results Aggregation -----------------------------------------------------------------
mean_loss = np.mean(results).round(4)
std_loss = np.std(results).round(4)
print(f'\nCross-Validation Results: {N_SPLITS}-fold')
print(f'Mean MSE: {mean_loss:.4f}')
print(f'Standard Deviation of MSE: {std_loss:.4f}')

# Visualization ------------------------------------------------------------------------
best_predictions = scaler.inverse_transform(best_predictions.reshape(-1, 1))

plt.plot(best_train_idx, data['Unemployment'][best_train_idx], label='Train')
plt.plot(best_test_idx[:-WINDOW-1], data['Unemployment'][best_test_idx][:-WINDOW-1], label='Test')
plt.plot(best_test_idx[:-WINDOW-1], best_predictions, label='Predictions')
plt.title(f'{MODEL} Forecast of Unemployment')
plt.xlabel('Time')
plt.ylabel('Unemployment')
plt.legend()
plt.show()

# Metrics -------------------------------------------------------------------------------
mse = mean_squared_error(data['Unemployment'][best_test_idx].values[:-WINDOW-1], best_predictions).round(4)
mae = mean_absolute_error(data['Unemployment'][best_test_idx].values[:-WINDOW-1], best_predictions).round(4)

print(f"\ncv_mean_loss: {mean_loss}")
print(f"\ncv_std_loss: {std_loss}")

data = {f'{MODEL}': [mean_loss, std_loss]}
index = ['mean_loss', 'std_loss']

results = pd.DataFrame(data, index=index).rename_axis('Metrics')

if not os.path.exists('results'):
    os.makedirs('results')

file_path = os.path.join('results', f'{MODEL}_results.csv')

results.to_csv(file_path)

# %%
