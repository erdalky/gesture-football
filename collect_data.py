import os
import cv2
import mediapipe as mp
import numpy as np
import time


ACTIONS = [
    "PASS_LEFT",
    "PASS_RIGHT",
    "SHOOT",
    "THROUGH_BALL",
    "HOLD"
]

SEQUENCE_LENGTH = 30
SAMPLES_PER_ACTION = 15
DATA_DIR = "data"


os.makedirs(DATA_DIR, exist_ok=True)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()


for action in ACTIONS:

    action_dir = os.path.join(DATA_DIR, action)
    os.makedirs(action_dir, exist_ok=True)

    print()
    print(f"Starting data collection for {action}")

    for sample_num in range(SAMPLES_PER_ACTION):

        sequence = []

        print(
            f"Get ready for {action} "
            f"sample {sample_num + 1}/{SAMPLES_PER_ACTION}"
        )

        time.sleep(1.5)

        while len(sequence) < SEQUENCE_LENGTH:

            success, frame = cap.read()

            if not success:
                print("Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                frame_features = []

                # Landmark 0 is the wrist
                wrist = hand_landmarks.landmark[0]

                # Keep global wrist position for movement information
                frame_features.extend([
                    wrist.x,
                    wrist.y,
                    wrist.z
                ])

# Store the other landmarks relative to the wrist
                for landmark in hand_landmarks.landmark[1:]:
                    frame_features.extend([
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])

                sequence.append(frame_features)

            cv2.putText(
                frame,
                f"{action} | "
                f"Sample {sample_num + 1}/{SAMPLES_PER_ACTION} | "
                f"Frames {len(sequence)}/{SEQUENCE_LENGTH}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Gesture Data Collection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                hands.close()
                cv2.destroyAllWindows()
                exit()

        if len(sequence) == SEQUENCE_LENGTH:

            sequence_array = np.array(
                sequence,
                dtype=np.float32
            )

            file_path = os.path.join(
                action_dir,
                f"{sample_num}.npy"
            )

            np.save(
                file_path,
                sequence_array
            )

            print(
                f"Saved {action} sample "
                f"{sample_num + 1}: "
                f"{sequence_array.shape}"
            )


cap.release()
hands.close()
cv2.destroyAllWindows()

print("Data collection complete.")