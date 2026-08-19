import cv2
import gradio as gr
import joblib
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    SIGN_CLASSES,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
    CONFIDENCE_THRESHOLD
)
from src.hand_detector import HandDetector
from src.text_decoder import TextDecoder


model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

detector = HandDetector()
decoder = TextDecoder()

sequence = deque(maxlen=SEQUENCE_LENGTH)


def predict(frame):
    if frame is None:
        return None, "Waiting for webcam...", "", 0.0

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    landmarks, processed_frame = detector.process_frame(frame)

    sequence.append(landmarks)

    sign = "Collecting..."
    confidence = 0.0

    if len(sequence) == SEQUENCE_LENGTH:
        data = np.array(sequence, dtype=np.float32)

        data = data.reshape(-1, FEATURES_PER_FRAME)
        data = scaler.transform(data)

        data = data.reshape(
            1,
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME
        )

        probabilities = model.predict(
            data,
            verbose=0
        )[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        if confidence >= CONFIDENCE_THRESHOLD:
            sign = SIGN_CLASSES[index]
            decoder.add_prediction(sign)
        else:
            sign = "UNKNOWN"

    output_frame = cv2.cvtColor(
        processed_frame,
        cv2.COLOR_BGR2RGB
    )

    text = decoder.get_text()

    return (
        output_frame,
        sign,
        text,
        round(confidence * 100, 2)
    )


def clear_text():
    decoder.clear()
    sequence.clear()

    return "", 0.0


with gr.Blocks(title="Sign Language to Text Converter") as demo:

    gr.Markdown(
        "# Sign Language to Text Converter"
    )

    gr.Markdown(
        "Real-time sign recognition using MediaPipe and LSTM."
    )

    with gr.Row():

        with gr.Column():
            camera = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Webcam"
            )

            output = gr.Image(
                label="Hand Detection"
            )

        with gr.Column():

            sign_output = gr.Textbox(
                label="Detected Sign"
            )

            text_output = gr.Textbox(
                label="Converted Text",
                lines=4
            )

            confidence_output = gr.Number(
                label="Confidence (%)"
            )

            clear_button = gr.Button(
                "Clear Text"
            )

    camera.stream(
        fn=predict,
        inputs=camera,
        outputs=[
            output,
            sign_output,
            text_output,
            confidence_output
        ],
        stream_every=0.1
    )

    clear_button.click(
        fn=clear_text,
        outputs=[
            text_output,
            confidence_output
        ]
    )


if __name__ == "__main__":
    demo.launch()