import json
import numpy as np

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SIGN_CLASSES,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME,
    X_PATH,
    Y_PATH,
    LABELS_PATH
)


def preprocess_data():
    X = []
    y = []

    for label, sign in enumerate(SIGN_CLASSES):
        sign_dir = RAW_DATA_DIR / sign

        if not sign_dir.exists():
            continue

        files = sorted(sign_dir.glob("*.npy"))

        for file_path in files:
            sequence = np.load(file_path)

            if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                continue

            X.append(sequence)
            y.append(label)

    if not X:
        raise ValueError("No valid dataset found.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(SIGN_CLASSES, file, indent=4)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {SIGN_CLASSES}")


if __name__ == "__main__":
    preprocess_data()