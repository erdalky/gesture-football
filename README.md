# Gesture Football

A real-time football control prototype that uses computer vision, deep learning, native C utilities, and a C++ game runtime to recognize hand gestures and trigger in-game actions.

## Overview

The project uses a multi-language architecture:

```text
Python
→ data collection
→ model training
→ ONNX export

C
→ low-level math
→ movement helpers
→ ball physics
→ collision utilities

C++
→ raylib football game
→ ONNX Runtime inference
→ calls C functions
→ gesture-controlled gameplay
```

The goal is to keep model development in Python, low-level reusable logic in C, and the real-time game/runtime layer in C++.

## Gesture Controls

| Gesture | Intended Action |
|---|---|
| `PASS_LEFT` | Pass left |
| `PASS_RIGHT` | Pass right |
| `SHOOT` | Shoot |
| `THROUGH_BALL` | Through ball |
| `HOLD` | Hold possession |

## Tech Stack

**Languages**
- Python
- C
- C++

**Machine Learning / Computer Vision**
- OpenCV
- MediaPipe
- NumPy
- PyTorch
- scikit-learn
- ONNX

**Native Runtime / Game**
- raylib
- ONNX Runtime
- clang / clang++
- macOS / Apple Silicon

## Machine Learning Pipeline

MediaPipe extracts 21 hand landmarks with `(x, y, z)` coordinates.

Each frame uses 63 features:

```text
3 global wrist coordinates
+
20 wrist-relative landmarks × 3
=
63 features
```

Each gesture is represented as a 30-frame sequence:

```text
Input shape: (30, 63)
```

The classifier is an LSTM with:

```text
Input size:   63
Hidden size:  64
Classes:      5
```

The current dataset contains 75 samples: 15 per gesture class.

## Native C Layer

C is used for low-level reusable utilities that do not depend on the game engine.

Planned C modules include:

```text
physics.c / physics.h
→ shot velocity
→ ball friction
→ movement math

collision.c / collision.h
→ field boundaries
→ ball/player collision helpers

vector_math.c / vector_math.h
→ vector normalization
→ distance
→ direction calculations
```

These functions are compiled as C and called from the C++ game through C-compatible headers.

Example:

```c
Vec2C calculate_shot_velocity(
    float angle_degrees,
    float speed
);
```

## C++ Game Runtime

The football prototype is written in C++17 using raylib.

Current functionality includes:

- WASD player movement
- 360-degree facing direction
- Ball possession
- Direction-based shooting
- Passing and through-ball actions
- Basic ball friction and reset logic
- ONNX Runtime model loading

Temporary keyboard controls are used while the gesture pipeline is being integrated.

## Model Deployment

The trained PyTorch model is saved as:

```text
gesture_lstm.pth
```

and exported to:

```text
gesture_lstm.onnx
```

The ONNX model is successfully loaded inside the C++ application using ONNX Runtime.

## Current Status

Completed:

- Webcam hand tracking
- Gesture dataset collection
- Wrist-relative landmark normalization
- PyTorch LSTM training
- Live Python gesture prediction
- ONNX export
- ONNX Runtime integration in C++
- raylib football prototype
- WASD movement
- 360-degree orientation
- Basic football action logic

## Next Steps

- Move physics and movement utilities into native C modules
- Run ONNX inference directly from C++
- Add webcam input to the C++ runtime
- Reproduce the Python preprocessing pipeline in C++
- Add confidence thresholds and gesture cooldowns
- Replace keyboard actions with gesture controls
- Add teammates, opponents, goals, scoring, and better ball physics
- Expand the dataset across more users and environments
- Measure inference latency, FPS, and false-trigger rate

## Build

Compile the C utilities:

```bash
clang -c physics.c -o physics.o
```

Build the C++ game and link the C object:

```bash
clang++ -std=c++17 game.cpp physics.o -o game \
-I$(brew --prefix raylib)/include \
-L$(brew --prefix raylib)/lib \
-Ithird_party/onnxruntime/include \
-L$(brew --prefix onnxruntime)/lib \
-lraylib \
-lonnxruntime \
-framework OpenGL \
-framework Cocoa \
-framework IOKit \
-framework CoreVideo
```

Run:

```bash
./game
```

## Goal

The final system will combine:

```text
Python ML training
+
C low-level utilities
+
C++ real-time game and inference
```

to create a gesture-controlled football game with a native runtime and deployable deep learning model.
