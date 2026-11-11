import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

print("Initializing PyTorch LSTM Temperature Forecaster training...")

# 1. Match architecture definition from main.py
class TemperatureLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(TemperatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 2. Load or generate sample data
try:
    df = pd.read_csv("weather_logs.csv")
except FileNotFoundError:
    np.random.seed(42)
    n_samples = 2000
    df = pd.DataFrame({
        'temperature_c': np.sin(np.linspace(0, 40, n_samples)) * 8 + 26,
        'humidity_pct': np.random.uniform(40, 90, n_samples),
        'pressure_hpa': np.random.uniform(1005, 1018, n_samples)
    })

# Extract feature matrix [Temperature, Humidity, Pressure]
features = df[['temperature_c', 'humidity_pct', 'pressure_hpa']].values

# 3. Create sliding sequences (Lookback = 24 steps)
SEQ_LEN = 24
X, y = [], []

for i in range(len(features) - SEQ_LEN):
    X.append(features[i : i + SEQ_LEN])
    y.append(features[i + SEQ_LEN, 0]) # Target: Temperature at step t+1

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# 4. Instantiate Model, Loss, and Optimizer
model = TemperatureLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
model.train()
EPOCHS = 20
print(f"Training LSTM over {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# 6. Save State Dict
torch.save(model.state_dict(), "lstm_temp_model.pt")
print("Success! PyTorch weights saved to lstm_temp_model.pt")