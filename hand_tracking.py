from collections import deque
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands (
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
SEQUENCE_LENGTH = 30
SEQUENCE_BUFFER = deque(maxlen=SEQUENCE_LENGTH)
cap = cv2.VideoCapture(0)

sequence_ready_printed = False 

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()  
    
print("Camera opened successfully.")

while True:
    success, frame = cap.read()
    
    if not success:
        print ("Error: Could not read frame.")
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            frame_features = []

            for i, landmark in enumerate(hand_landmarks.landmark): 
                frame_features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            SEQUENCE_BUFFER.append(frame_features)
            
            if ( 
                len(SEQUENCE_BUFFER) == SEQUENCE_LENGTH and
                not sequence_ready_printed
            ):
                print ("Sequence ready: (30, 63)")
                sequence_ready_printed = True  # Set the flag to True after printing
                sequence_array = np.array(SEQUENCE_BUFFER)
                
    cv2.imshow("Gesture Football - Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
hands.close()
cv2.destroyAllWindows()
print ("Camera released and all windows closed.")