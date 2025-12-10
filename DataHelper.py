import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)


def data_generator(images, steering_angles, batch_size=32):
    """
    Create shuffled mini-batches for training and validation.

    Parameters:
        images: NumPy array of images, shape (N, H, W, 3)
        steering_angles: NumPy array of steering angles, shape (N,)
        batch_size: size of each mini-batch

    Yields:
        X_batch: float32 images, shape (B, H, W, 3)
        y_batch: float32 angles, shape (B,)
    """
    images = np.asarray(images)
    steering_angles = np.asarray(steering_angles, dtype=np.float32)
    num_samples = len(images)

    while True:
        indices = np.random.permutation(num_samples)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            batch_images = images[batch_indices]
            batch_steering = steering_angles[batch_indices]

            X_batch = batch_images.astype(np.float32)
            y_batch = batch_steering.astype(np.float32)

            yield X_batch, y_batch


def split_data(
    images,
    steering_angles,
    test_size=0.2,
    random_state=42,
):
    """
    Split images and angles into train and validation sets.

    Returns:
        train_images, val_images, train_angles, val_angles
    """
    return train_test_split(
        images,
        steering_angles,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def create_callbacks(best_model_path="best_model.h5", patience=3):
    """
    Create a list of callbacks for stable training.

    Includes:
        ModelCheckpoint   save best model on validation loss
        EarlyStopping     stop if no progress
        ReduceLROnPlateau lower learning rate if stuck
    """
    checkpoint_cb = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience,
        min_lr=1e-6,
        verbose=1,
    )

    return [checkpoint_cb, early_stop_cb, reduce_lr_cb]


def train_with_generators(
    model,
    train_images,
    train_angles,
    val_images,
    val_angles,
    batch_size=32,
    epochs=5,
    callbacks=None,
):
    """
    Train a model with generators and return history.

    Parameters:
        model: compiled Keras model
        train_images: training images
        train_angles: training steering angles
        val_images: validation images
        val_angles: validation steering angles
        batch_size: batch size
        epochs: number of epochs
        callbacks: list of Keras callbacks

    Returns:
        model: trained model
        history: Keras History object
    """
    train_gen = data_generator(train_images, train_angles, batch_size)
    val_gen = data_generator(val_images, val_angles, batch_size)

    steps_per_epoch = ceil(len(train_images) / batch_size)
    val_steps = ceil(len(val_images) / batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def plot_history(history, out_path=None):
    """
    Plot training and validation loss over epochs.

    Parameters:
        history: Keras History object
        out_path: optional path to save the plot
    """
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Train loss")
    plt.plot(epochs_range, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
