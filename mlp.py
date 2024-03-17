import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, dropout_prob=0.2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 2)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)  # Apply Dropout
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)  # Apply Dropout
        x = self.fc_out(x)
        return x
