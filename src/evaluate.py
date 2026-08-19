import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.keras.models import load_model

from config import (
    X_PATH,
    Y_PATH,
    MODEL_PATH,
    SCALER_PATH,
    RESULTS_DIR,
    SIGN_CLASSES
)


def evaluate_model():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    scaler = joblib.load(SCALER_PATH)

    samples, frames, features = X.shape

    X = X.reshape(-1, features)
    X = scaler.transform(X)
    X = X.reshape(samples, frames, features)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = load_model(MODEL_PATH)

    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print()

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(SIGN_CLASSES))),
        target_names=SIGN_CLASSES,
        zero_division=0
    )

    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(SIGN_CLASSES)))
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=SIGN_CLASSES,
        yticklabels=SIGN_CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Sign Language Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )
    plt.close()


if __name__ == "__main__":
    evaluate_model()