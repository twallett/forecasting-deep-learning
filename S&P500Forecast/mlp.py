#%%
from utils import *
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

# Hyperparameters ---------------------------------------------------------------------

SEED = 123

TICKER = 'SPY'
START = '2014-01-01'
END = '2024-01-01'

ASPLIT = 0.8

WINDOW = 7

BATCH_SIZE = 64

HIDDEN_DIM = 16
OUTPUT_DIM = 1
LR = 1e-02

EPOCHS = 150

# Reproducibility ---------------------------------------------------------------------

seed_everything(SEED)

# Load Data ---------------------------------------------------------------------------
data = yf.download(TICKER, 
                   start = START, 
                   end = END)

# Scale Data --------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split Data --------------------------------------------------------------------------
split = int(len(scaled_data) * ASPLIT)
train = scaled_data[:split]
test = scaled_data[split:]

# PyTorch Dataset ---------------------------------------------------------------------
class Dataset(Dataset):
    def __init__(self, data, window=1):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window - 1

    def __getitem__(self, idx):
        X = self.data[idx:(idx + self.window), 0]
        Y = self.data[idx + self.window, 0]
        return torch.tensor(X).float(), torch.tensor(Y).float()
    
train_dataset = Dataset(train,
                        window = WINDOW)
test_dataset = Dataset(test,
                       window = WINDOW)

# PyTorch Dataloader ------------------------------------------------------------------
train_loader = DataLoader(train_dataset,
                          batch_size = BATCH_SIZE)
test_loader = DataLoader(test_dataset,
                         batch_size = BATCH_SIZE)

# LSTM Model --------------------------------------------------------------------------
class Forecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Forecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.sig(self.fc1(x)))

model = Forecast(input_dim = WINDOW, 
                 hidden_dim = HIDDEN_DIM, 
                 output_dim = OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train-Test Loop ----------------------------------------------------------------------
predictions = []
for epoch in range(EPOCHS):
    print(f"epoch: {epoch}")
    for inputs, targets in tqdm(train_loader):
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
            
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:        
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss
            
            if epoch == EPOCHS - 1:
                predictions.append(outputs)
    print(f"test loss: { (total_loss / len(test_loader)).item().__round__(4) }")
    
# Visualization ------------------------------------------------------------------------
predictions = scaler.inverse_transform(np.concatenate(predictions))

plt.plot(range(0, len(train)), data['Close'][:split], label = 'Train')
plt.plot(range(len(train), len(train) + len(test) - WINDOW - 1), data['Close'][split:-WINDOW-1], label = 'Test')
plt.plot(range(len(train), len(train) + len(test) - WINDOW - 1), predictions, label = 'Predictions')
plt.title('MLP Forecast of SPY')
plt.xlabel('Time')
plt.ylabel('SPY Close')
plt.legend()
plt.show()

print(f"\n mse: {mean_squared_error(data['Close'][split:-WINDOW-1].values,predictions).round(4)}")
print(f"\n mae: {mean_absolute_error(data['Close'][split:-WINDOW-1].values,predictions).round(4)}")

# %%
