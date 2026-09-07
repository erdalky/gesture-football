import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

ACTIONS = [
    "PASS_LEFT",
    "PASS_RIGHT",
    "SHOOT",
    "THROUGH_BALL",
    "HOLD"
]

DATA_DIR = "data"

X = []
y = []

for label, action in enumerate(ACTIONS):

    action_dir = os.path.join(DATA_DIR, action)

    for file_name in os.listdir(action_dir):

        if file_name.endswith(".npy"):

            file_path = os.path.join(
                action_dir,
                file_name
            )

            sequence = np.load(file_path)

            X.append(sequence)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)

print("Training labels:", y_train)
print("Test labels:", y_test)

class GestureLSTM(nn.Module):

    def __init__(
        self,
        input_size=63,
        hidden_size=64,
        num_classes=5
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        lstm_output, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]

        output = self.fc(final_hidden)

        return output
    
model = GestureLSTM()

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 100

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {loss.item():.4f}"
        )
        
model.eval()

with torch.no_grad():

    test_outputs = model(X_test)

    predictions = torch.argmax(
        test_outputs,
        dim=1
    )

    correct = (predictions == y_test).sum().item()

    accuracy = correct / len(y_test)

print("Predictions:", predictions)
print("Actual labels:", y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

print(f"Test accuracy: {accuracy * 100:.2f}%")

torch.save(
    model.state_dict(),
    "gesture_lstm.pth"
)

print("Model saved to gesture_lstm.pth")