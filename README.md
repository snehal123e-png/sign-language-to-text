# Sign Language to Text Converter

Stage 3 Hard-level project using Python, MediaPipe, LSTM, TensorFlow, and Gradio.

## Pipeline
Webcam → MediaPipe Hand Landmarks → Frame Sequence → LSTM → Sign Prediction → Text Decoder → Gradio UI

## Sign Classes
HELLO, THANK_YOU, YES, NO, PLEASE, HELP, GOOD, BAD, I, YOU

## Project Structure
- `dataset/` — raw and processed training data
- `models/` — trained model and preprocessing artifacts
- `src/` — data collection, preprocessing, training, evaluation, prediction, and text decoding
- `results/` — evaluation outputs
- `app.py` — Gradio application
