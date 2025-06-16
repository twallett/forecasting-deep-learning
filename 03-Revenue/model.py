import torch
import torch.nn as nn

class GRUForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        out, _ = self.gru(x.unsqueeze(2), (h0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        out, _ = self.lstm(x.unsqueeze(2), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class RNNForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.fc(self.rnn(x)[0])
    
class MLPForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.sig(self.fc1(x)))
    
class Forecast(nn.Module):
    def __init__(self, model, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        if model == 'MLP':
            self.ff = MLPForecast(input_dim, hidden_dim, output_dim)
        elif model == 'RNN':
            self.ff = RNNForecast(input_dim, hidden_dim, output_dim)
        elif model == 'LSTM':
            self.ff = LSTMForecast(input_dim, hidden_dim, num_layers, output_dim)
        else:
            self.ff = GRUForecast(input_dim, hidden_dim, num_layers, output_dim)
    
    def forward(self, x):
        return self.ff(x)
        