import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import (
    MAX_HANDS,
    FEATURES_PER_FRAME,
    MODELS_DIR
)

class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=str(MODELS_DIR / "hand_landmarker.task")
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        landmarks = np.zeros(
            FEATURES_PER_FRAME,
            dtype=np.float32
        )

        if result.hand_landmarks:
            for hand_index, hand in enumerate(
                result.hand_landmarks[:MAX_HANDS]
            ):
                start = hand_index * 63

                for landmark_index, landmark in enumerate(hand):
                    offset = start + landmark_index * 3

                    landmarks[offset] = landmark.x
                    landmarks[offset + 1] = landmark.y
                    landmarks[offset + 2] = landmark.z

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1
                    )

        return landmarks, frame

    def close(self):
        self.detector.close()