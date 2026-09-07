# Gesture Football

A real-time football control prototype that uses computer vision and deep learning to recognize hand gestures and trigger in-game actions.

## Overview

The project combines a Python-based ML pipeline with a C++ game runtime:

```text
Webcam
→ MediaPipe hand landmarks
→ 30-frame gesture sequence
→ PyTorch LSTM
→ ONNX export
→ ONNX Runtime in C++
→ raylib football game
```

## Gesture Controls

| Gesture | Action |
|---|---|
| `PASS_LEFT` | Open palm swipe left |
| `PASS_RIGHT` | Open palm swipe right |
| `SHOOT` | Closed fist moved upward |
| `THROUGH_BALL` | Index finger extended and held still |
| `HOLD` | Open palm held still |

## Tech Stack

**ML / Computer Vision**
- Python
- OpenCV
- MediaPipe
- NumPy
- PyTorch
- scikit-learn
- ONNX

**Game / Runtime**
- C++
- raylib
- ONNX Runtime
- clang++

## Model

MediaPipe provides 21 hand landmarks with `(x, y, z)` coordinates.

Each frame is represented by 63 features:

```text
3 global wrist coordinates
+
20 wrist-relative landmarks × 3
=
63 features
```

Each gesture contains 30 frames, producing an input shape of:

```text
(30, 63)
```

The classifier is an LSTM with:

```text
Input size:   63
Hidden size:  64
Classes:      5
```

The current dataset contains 75 samples: 15 per gesture class.

## Current Status

Completed:

- Webcam hand tracking
- Landmark extraction and normalization
- Gesture dataset collection
- PyTorch LSTM training
- Live gesture prediction in Python
- ONNX model export
- ONNX Runtime integration in C++
- raylib football prototype
- WASD movement and 360° facing direction
- Basic pass, shoot, through-ball, and hold logic

The latest small held-out test reached **100% accuracy**, but more diverse data is still needed to evaluate real-world generalization.

## Running the Project

Activate the environment:

```bash
conda activate gesture-football
```

Train the model:

```bash
python train_model.py
```

Test live prediction:

```bash
python predict_live.py
```

Export to ONNX:

```bash
python export_onnx.py
```

Build the C++ game:

```bash
clang++ -std=c++17 game.cpp -o game \
-I$(brew --prefix raylib)/include \
-L$(brew --prefix raylib)/lib \
-Ithird_party/onnxruntime/include \
-L$(brew --prefix onnxruntime)/lib \
-lraylib -lonnxruntime -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
```

Run:

```bash
./game
```

## Next Steps

- Run ONNX inference directly from C++
- Add webcam input to the C++ runtime
- Recreate the same preprocessing pipeline in C++
- Add confidence thresholds and gesture cooldowns
- Replace keyboard actions with gesture controls
- Add teammates, opponents, goals, scoring, and better ball physics
- Expand the dataset across different sessions, lighting conditions, distances, and users
- Measure inference latency, FPS, and false-trigger rate

## Goal

The final system will run the football game natively in C++ while recognizing hand gestures in real time to control football actions.
