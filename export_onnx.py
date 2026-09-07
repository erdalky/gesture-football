import torch
import torch.nn as nn


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
        _, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        return self.fc(final_hidden)


print("Loading PyTorch model...")

model = GestureLSTM()

model.load_state_dict(
    torch.load(
        "gesture_lstm.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded.")


dummy_input = torch.randn(
    1,
    30,
    63
)

print("Exporting to ONNX...")


torch.onnx.export(
    model,
    dummy_input,
    "gesture_lstm.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)


print("Model exported to gesture_lstm.onnx")