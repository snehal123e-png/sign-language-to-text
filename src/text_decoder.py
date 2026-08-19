import cv2
import numpy as np
import joblib
from collections import deque
from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH,
    SCALER_PATH,
    SIGN_CLASSES,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
    CONFIDENCE_THRESHOLD,
    PREDICTION_SMOOTHING
)

from hand_detector import HandDetector
from text_decoder import TextDecoder


class SignPredictor:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.detector = HandDetector()

        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(
            maxlen=PREDICTION_SMOOTHING
        )

        self.decoder = TextDecoder()

    def predict_sequence(self, sequence):
        data = np.array(sequence, dtype=np.float32)

        data = data.reshape(
            -1,
            FEATURES_PER_FRAME
        )

        data = self.scaler.transform(data)

        data = data.reshape(
            1,
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME
        )

        prediction = self.model.predict(
            data,
            verbose=0
        )[0]

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_index]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            return "UNKNOWN", confidence

        return (
            SIGN_CLASSES[class_index],
            confidence
        )

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            landmarks, frame = (
                self.detector.process_frame(frame)
            )

            self.sequence.append(landmarks)

            sign = "Collecting..."
            confidence = 0.0

            if len(self.sequence) == SEQUENCE_LENGTH:

                sign, confidence = (
                    self.predict_sequence(
                        list(self.sequence)
                    )
                )

                if sign != "UNKNOWN":

                    self.prediction_history.append(
                        sign
                    )

                    self.decoder.add_prediction(
                        sign
                    )

            text = self.decoder.get_text()

            cv2.putText(
                frame,
                f"Sign: {sign}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {len(self.sequence)}/{SEQUENCE_LENGTH}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "C = Clear Text | Q = Quit",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Text: {text}",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                "Sign Language Prediction",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                self.decoder.clear()

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.detector.close()


if __name__ == "__main__":
    predictor = SignPredictor()
    predictor.run()