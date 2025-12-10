import numpy as np
from math import ceil


def data_generator(images, steering_angles, batch_size=32):
    """
    Generate shuffled mini-batches for training or validation.

    Parameters:
        images: NumPy array of images, shape (N, H, W, 3)
        steering_angles: NumPy array of angles, shape (N,)
        batch_size: size of each mini-batch

    Yields:
        X_batch: float32 images, shape (B, H, W, 3)
        y_batch: float32 angles, shape (B,)
    """
    images = np.asarray(images)
    steering_angles = np.asarray(steering_angles, dtype=np.float32)
    num_samples = len(images)

    while True:
        # shuffle indices each epoch
        indices = np.random.permutation(num_samples)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]

            batch_images = images[batch_indices]
            batch_steering = steering_angles[batch_indices]

            X_batch = batch_images.astype(np.float32)
            y_batch = batch_steering.astype(np.float32)

            yield X_batch, y_batch


def train_model(
    model,
    train_images,
    train_steering_angles,
    validation_images,
    validation_steering_angles,
    batch_size=32,
    epochs=5,
):
    """
    Train the model with generators.

    Parameters:
        model: compiled Keras model
        train_images: training images
        train_steering_angles: training angles
        validation_images: validation images
        validation_steering_angles: validation angles
        batch_size: mini-batch size
        epochs: number of epochs

    Returns:
        trained model
    """
    train_generator = data_generator(
        train_images, train_steering_angles, batch_size
    )
    validation_generator = data_generator(
        validation_images, validation_steering_angles, batch_size
    )

    steps_per_epoch = ceil(len(train_images) / batch_size)
    validation_steps = ceil(len(validation_images) / batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
    )
    return model, history


def create_model():
    """
    Create a CNN model based on the NVIDIA architecture.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        Dense,
        Flatten,
        Dropout,
        Lambda,
    )

    model = Sequential()

    # Normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))

    # Convolution layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    # Flatten
    model.add(Flatten())

    # Fully connected layers with dropout for regularization
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))

    # Output layer
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model


if __name__ == "__main__":
    from data import load_driving_log, preprocess_data, augment_data
    from sklearn.model_selection import train_test_split

    # Load and preprocess data
    df = load_driving_log()
    images, steering_angles = preprocess_data(df)

    # Augment data
    images, steering_angles = augment_data(images, steering_angles)

    # Split into train and validation
    train_images, validation_images, train_steering_angles, validation_steering_angles = train_test_split(
        images,
        steering_angles,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # Create model
    model = create_model()

    # Train model
    model, history = train_model(
        model,
        train_images,
        train_steering_angles,
        validation_images,
        validation_steering_angles,
        batch_size=32,
        epochs=5,
    )

    # Save model
    model.save("model.h5")
