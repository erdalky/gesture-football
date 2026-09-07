import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque


ACTIONS = [
    "PASS_LEFT",
    "PASS_RIGHT",
    "SHOOT",
    "THROUGH_BALL",
    "HOLD"
]

SEQUENCE_LENGTH = 30


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

model.load_state_dict(
    torch.load(
        "gesture_lstm.pth",
        map_location="cpu"
    )
)

model.eval()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction_text = "Collecting frames..."
    confidence_text = ""

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        wrist = hand_landmarks.landmark[0]

        frame_features = []

        # Global wrist position
        frame_features.extend([
            wrist.x,
            wrist.y,
            wrist.z
        ])

        # Relative landmark positions
        for landmark in hand_landmarks.landmark[1:]:

            frame_features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        sequence.append(
            frame_features
        )

        if len(sequence) == SEQUENCE_LENGTH:

            input_sequence = np.array(
                sequence,
                dtype=np.float32
            )

            input_tensor = torch.tensor(
                input_sequence,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                logits = model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = prediction.item()
            confidence_value = confidence.item()

            prediction_text = ACTIONS[
                predicted_class
            ]

            confidence_text = (
                f"{confidence_value * 100:.1f}%"
            )

    else:

        sequence.clear()

        prediction_text = "No hand detected"


    cv2.putText(
        frame,
        prediction_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Live Gesture Prediction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()