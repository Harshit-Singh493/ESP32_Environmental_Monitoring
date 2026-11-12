import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("Initializing PyTorch LSTM Temperature Forecaster training...")


# 1. Match architecture definition from main.py
class TemperatureLSTM(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(TemperatureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 2. Load actual logged weather data (fail loudly if missing)
df = pd.read_csv("weather_logs.csv")
features = df[["temperature_c", "humidity_pct", "pressure_hpa"]].dropna().values

# 3. Scale features to mean=0, std=1 and save scaler for main.py
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
joblib.dump(scaler, "lstm_scaler.joblib")

# 4. Create sliding sequences (Lookback = 24 steps)
SEQ_LEN = 24
X, y = [], []

for i in range(len(scaled_features) - SEQ_LEN):
    X.append(scaled_features[i : i + SEQ_LEN])
    y.append(scaled_features[i + SEQ_LEN, 0])  # Scaled temperature at t+1

X_tensor = torch.tensor(np.array(X, dtype=np.float32))
y_tensor = torch.tensor(np.array(y, dtype=np.float32).reshape(-1, 1))

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Instantiate Model, Loss, and Optimizer
model = TemperatureLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Mini-Batch Training Loop
model.train()
EPOCHS = 40
print(f"Training LSTM over {EPOCHS} epochs on {len(dataset)} sequences...")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_X)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}], MSE Loss: {epoch_loss / len(dataset):.4f}"
        )

# 7. Save State Dict
torch.save(model.state_dict(), "lstm_temp_model.pt")
print("Saved lstm_temp_model.pt and lstm_scaler.joblib successfully.")