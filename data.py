import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def load_driving_log(csv_path: str = "driving_log.csv") -> pd.DataFrame:
    """
    Load the driving log CSV file and keep only center image path and steering angle.

    Parameters:
        csv_path: Path to the driving_log.csv file.

    Returns:
        DataFrame with columns ['Center', 'Steering'].
    """
    df = pd.read_csv(csv_path, header=None)
    df.rename(
        columns={
            0: "Center",
            1: "Left",
            2: "Right",
            3: "Steering",
            4: "Throttle",
            5: "Brake",
            6: "Speed",
        },
        inplace=True,
    )

    return df[["Center", "Steering"]]


def plot_steering_histogram(df: pd.DataFrame) -> None:
    """
    Plot a histogram of the steering angles.

    Parameters:
        df: DataFrame containing a 'Steering' column.
    """
    plt.hist(df["Steering"], bins=50, edgecolor="black")
    plt.title("Histogram of Steering Angles")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def preprocess_data(df: pd.DataFrame, image_dir: str = "images"):
    """
    Load and preprocess center camera images.

    Steps:
    - Resolve file name from path.
    - Read image from disk.
    - Crop sky and hood.
    - Convert to YUV.
    - Resize to network input resolution.

    Parameters:
        df: DataFrame with columns ['Center', 'Steering'].
        image_dir: Directory where images are stored.

    Returns:
        images: NumPy array of shape (N, H, W, 3) in YUV.
        steering_angles: NumPy array of shape (N,).
    """
    image_paths = df["Center"].astype(str).values
    steering_angles = df["Steering"].values.astype(np.float32)

    images = []
    valid_angles = []

    for path, angle in zip(image_paths, steering_angles):
        filename = os.path.basename(path.strip())
        full_path = os.path.join(image_dir, filename)

        image = cv2.imread(full_path)
        if image is None:
            continue

        image = image[60:135, :, :]  # shape about (75, W, 3)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        image = cv2.resize(image, (66, 200))

        images.append(image)
        valid_angles.append(angle)

    images = np.asarray(images, dtype=np.uint8)
    steering = np.asarray(valid_angles, dtype=np.float32)
    return images, steering


def augment_data(images: np.ndarray, steering_angles: np.ndarray):
    """
    Apply random data augmentation.

    Augmentations:
    - Random brightness change.
    - Random translation in x and y, with steering adjustment.
    - Random horizontal flip, with steering sign inversion.

    Parameters:
        images: NumPy array of shape (N, H, W, 3) in YUV.
        steering_angles: NumPy array of shape (N,).

    Returns:
        augmented_images: NumPy array of augmented images.
        augmented_steering_angles: NumPy array of augmented steering angles.
    """
    images = np.asarray(images)
    steering_angles = np.asarray(steering_angles, dtype=np.float32)

    augmented_images = []
    augmented_steering_angles = []

    for image, angle in zip(images, steering_angles):
        img = image.copy()
        steering = float(angle)

        if np.random.rand() < 0.5:
            bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            random_bright = 0.25 + np.random.uniform()
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random_bright, 0, 255)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        if np.random.rand() < 0.5:
            translation_x = 100 * (np.random.rand() - 0.5)
            translation_y = 10 * (np.random.rand() - 0.5)
            steering += translation_x * 0.002

            translation_matrix = np.float32(
                [[1, 0, translation_x], [0, 1, translation_y]]
            )
            img = cv2.warpAffine(
                img,
                translation_matrix,
                (img.shape[1], img.shape[0]),
                borderMode=cv2.BORDER_REPLICATE,
            )

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            steering = -steering

        augmented_images.append(img)
        augmented_steering_angles.append(steering)

    augmented_images = np.asarray(augmented_images, dtype=np.uint8)
    augmented_steering_angles = np.asarray(
        augmented_steering_angles, dtype=np.float32
    )

    return augmented_images, augmented_steering_angles


def prepare_data(csv_path: str = "driving_log.csv", image_dir: str = "images"):
    """
    Full pipeline:
    - Load CSV.
    - Preprocess images.
    - Augment dataset.

    Returns:
        X: NumPy array of augmented images.
        y: NumPy array of augmented steering angles.
    """
    df = load_driving_log(csv_path)
    images, steering_angles = preprocess_data(df, image_dir=image_dir)
    aug_images, aug_steering_angles = augment_data(images, steering_angles)
    return aug_images, aug_steering_angles
