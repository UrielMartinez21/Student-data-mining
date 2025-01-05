import torch
import torch.nn as nn
import torch.optim as optim


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)    # Capa oculta 1
        self.fc2 = nn.Linear(64, 32)            # Capa oculta 2
        self.fc3 = nn.Linear(32, 16)            # Capa oculta 3
        self.fc4 = nn.Linear(16, 1)             # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))          # Activación ReLU en capa 1
        x = torch.relu(self.fc2(x))          # Activación ReLU en capa 2
        x = torch.relu(self.fc3(x))          # Activación ReLU en capa 3
        x = self.fc4(x)                      # Salida continua
        return x


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss