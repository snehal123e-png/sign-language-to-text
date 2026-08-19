import numpy as np
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import (
    X_PATH,
    Y_PATH,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    NUM_CLASSES,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    FEATURES_PER_FRAME
)


def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    return X, y


def scale_data(X):
    samples, frames, features = X.shape

    scaler = StandardScaler()

    X_reshaped = X.reshape(-1, features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(samples, frames, features)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled


def build_model(input_shape):
    model = Sequential([
        LSTM(
            LSTM_UNITS_1,
            return_sequences=True,
            input_shape=input_shape
        ),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(
            LSTM_UNITS_2,
            return_sequences=False
        ),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(64, activation="relu"),
        Dropout(DROPOUT_RATE),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    X, y = load_data()

    X = scale_data(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )

    model = build_model(
        input_shape=(
            X_train.shape[1],
            FEATURES_PER_FRAME
        )
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    train()