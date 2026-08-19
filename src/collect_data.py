import cv2
import numpy as np
import time

from config import (
    RAW_DATA_DIR,
    SIGN_CLASSES,
    SEQUENCE_LENGTH,
    SEQUENCES_PER_CLASS,
    COLLECTION_DELAY
)

from hand_detector import HandDetector


def collect_data():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    for sign in SIGN_CLASSES:

        sign_dir = RAW_DATA_DIR / sign
        sign_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(sign_dir.glob("sequence_*.npy"))
        sequence_count = len(existing_files)

        if sequence_count >= SEQUENCES_PER_CLASS:
            print(f"[DONE] {sign}: {sequence_count}/{SEQUENCES_PER_CLASS}")
            continue

        print()
        print("=" * 50)
        print(f"CURRENT SIGN: {sign}")
        print(f"Existing sequences: {sequence_count}")
        print(f"Remaining: {SEQUENCES_PER_CLASS - sequence_count}")
        print("=" * 50)

        while sequence_count < SEQUENCES_PER_CLASS:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            landmarks, frame = detector.process_frame(frame)

            cv2.putText(
                frame,
                f"Sign: {sign}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Sequence: {sequence_count + 1}/{SEQUENCES_PER_CLASS}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Record | Q = Quit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Sign Language Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                return

            if key == 32:

                sequence = []

                print(
                    f"Recording {sign} "
                    f"sequence {sequence_count + 1}..."
                )

                for _ in range(SEQUENCE_LENGTH):

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)

                    landmarks, frame = detector.process_frame(
                        frame
                    )

                    sequence.append(landmarks)

                    cv2.putText(
                        frame,
                        f"Recording: {sign}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Frame: {len(sequence)}/{SEQUENCE_LENGTH}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.imshow(
                        "Sign Language Data Collection",
                        frame
                    )

                    cv2.waitKey(1)

                    time.sleep(COLLECTION_DELAY)

                if len(sequence) == SEQUENCE_LENGTH:

                    sequence_path = (
                        sign_dir /
                        f"sequence_{sequence_count:04d}.npy"
                    )

                    np.save(
                        sequence_path,
                        np.array(sequence, dtype=np.float32)
                    )

                    sequence_count += 1

                    print(
                        f"Saved: {sequence_path}"
                    )

        print(f"[COMPLETE] {sign}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print()
    print("=" * 50)
    print("DATA COLLECTION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    collect_data()