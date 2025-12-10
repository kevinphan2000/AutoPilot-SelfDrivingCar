import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# load csv from driving_log.csv
def load_driving_log(csv_path='driving_log.csv'):
    """
    Load the driving log CSV file into a pandas DataFrame.

    Parameters:
    csv_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the driving log data.
    """
    # driving log has seven column: 
    # Center, Left, Right, Steering, Throttle, Brake, Speed
    # Use only Center and Steering for training

    df = pd.read_csv(csv_path, header=None)
    # columns are unamed, so rename them
    df.rename(columns={0: 'Center', 1: 'Left', 2: 'Right', 3: 'Steering', 4: 'Throttle', 5: 'Brake', 6: 'Speed'}, inplace=True)
    # return only Center and Steering columns
    
    return df[['Center', 'Steering']]



#plot histogram of steering angles
def plot_steering_histogram(df):
    """
    Plot a histogram of the steering angles.

    Parameters:
    df (pd.DataFrame): DataFrame containing the driving log data.
    """
    

    plt.hist(df['Steering'], bins=50, edgecolor='black')
    plt.title('Histogram of Steering Angles')
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Preprocess data
# creating training data and labels
def preprocess_data(df):
    """
    Preprocess the images and steering angles from the DataFrame.
    
    :param df: Description
    """
    # Extract image paths and steering angles
    image_paths = df['Center'].values
    steering_angles = df['Steering'].values
    # Load image as numpy array
    for i in range(len(image_paths)):
        # trim path to just the filename
        image_paths[i] = image_paths[i].split('\\')[-1]
        # load image in images folder
        image = cv2.imread('images/' + image_paths[i])
        # crop the image (full width, [60-135px] height)
        image = image[60:135, :, :]
        # apply Gaussian blur
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # convert image to YUV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # resize image to 200x66 (as per NVIDIA model input)
        image = cv2.resize(image, (200, 66))
        # normalize to 0-1 range and convert to float32
        image = image.astype(np.float32) / 255.0
        image_paths[i] = image
    return image_paths, steering_angles


# data augmentation: randomly apply brightness, shift, flip to certain amount of images
def augment_data(images, steering_angles):
    """
    Augment the dataset by applying random transformations to the images.
    
    :param images: Description
    :param steering_angles: Description
    """
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(images, steering_angles):
        # Ensure image is float for augmentation
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add original image
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        
        # ALWAYS flip to balance left/right turns
        flipped_image = cv2.flip(image, 1)
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(-steering_angle)
        
        # Randomly apply brightness to both
        if np.random.rand() < 0.5:
            # Convert to BGR then HSV for brightness adjustment
            image_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            random_bright = .25 + np.random.uniform()
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random_bright, 0, 255)
            image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0
            augmented_images.append(image)
            augmented_steering_angles.append(steering_angle)
        
        # Randomly shift image
        if np.random.rand() < 0.5:
            translation_x = 100 * (np.random.rand() - 0.5)
            translation_y = 10 * (np.random.rand() - 0.5)  
            steering_angle += translation_x * 0.002
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        
        # Randomly flip image
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle

        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
    return augmented_images, augmented_steering_angles


# to be called from model.py to get training data and labels
def prepare_data() -> tuple[np.ndarray, np.ndarray]:
    df = load_driving_log()
    # plot_steering_histogram(df)
    images, steering_angles = preprocess_data(df)
    # aug_images, aug_steering_angles = augment_data(images, steering_angles)
    
    return np.array(images), np.array(steering_angles)
    

    
