import cv2
import mediapipe as mp

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Model configuration
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

# Create hand landmarker
landmarker = HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened.")
    landmarker.close()
    exit()

print("✅ Camera started!")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Could not read camera frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = landmarker.detect(mp_image)

    # Draw landmarks
    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            points = []

            # Get 21 hand landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Hand connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17), (0, 17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

    # Display
    cv2.imshow(
        "Sign Language - Hand Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
landmarker.close()
cv2.destroyAllWindows()