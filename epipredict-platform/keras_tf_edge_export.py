"""
keras_tf_edge_export.py
Builds the EpiPredict Multi-Level CNN + Bidirectional LSTM architecture in TensorFlow/Keras
and exports an 8-bit quantized TFLite model for low-latency wearable edge deployment.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_epipredict_keras_model(
    num_channels: int = 18,
    num_bands: int = 6,
    time_bins: int = 32,
    num_classes: int = 2,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(num_channels, num_bands, time_bins), name="dwt_scalogram_input")

    # Permute to (Batch, Bands, TimeBins, Channels) for 2D Spatial-Spectral Convolution
    x = layers.Permute((2, 3, 1))(inputs)

    # Multi-Level Convolutional Feature Extractor
    x = layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Channel Attention Squeeze-and-Excitation Gate
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(8, activation="relu")(se)
    se = layers.Dense(32, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, 32))(se)
    x = layers.Multiply()([x, se])

    # Reshape along TimeBins axis for Bidirectional LSTM Temporal Modeling
    x = layers.Permute((2, 1, 3))(x)  # (Batch, TimeBins, Bands, 32)
    x = layers.Reshape((time_bins, num_bands * 32))(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2))(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="preictal_risk_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EpiPredict_CNN_BiLSTM_Keras")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_quantized_tflite(model: tf.keras.Model, export_path: str = "epipredict_edge_int8.tflite") -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"[TensorFlow/Keras Exporter] Saved quantized edge model ({len(tflite_model) / 1024:.1f} KB) -> {export_path}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    keras_model = build_epipredict_keras_model()
    keras_model.summary()

    # Verify forward pass with synthetic CHB-MIT wavelet batch
    dummy_x = np.random.randn(4, 18, 6, 32).astype(np.float32)
    preds = keras_model.predict(dummy_x, verbose=0)
    print(f"[TensorFlow/Keras] Output Probability Shape: {preds.shape}, Sample Pre-Ictal Risk: {preds[0, 1]*100:.2f}%")
    export_quantized_tflite(keras_model)