
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def load_driving_log(csv_path='driving_log.csv'):
    """
    Load the driving log CSV file into a pandas DataFrame.

    Parameters:
    csv_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the driving log data.
    """

    df = pd.read_csv(csv_path, header=None)
    df.rename(columns={0: 'Center', 1: 'Left', 2: 'Right', 3: 'Steering', 4: 'Throttle', 5: 'Brake', 6: 'Speed'}, inplace=True)
    
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


def preprocess_data(df):
    """
    Docstring for preprocess_data
    
    :param df: Description
    """
    image_paths = df['Center'].values
    steering_angles = df['Steering'].values
    for i in range(len(image_paths)):
        image_paths[i] = image_paths[i].split('\\')[-1]
        image = cv2.imread('images/' + image_paths[i])
        image = image[60:135, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (66, 200))
        image_paths[i] = image
    return image_paths, steering_angles


def augment_data(images, steering_angles):
    """
    Docstring for augment_data
    
    :param images: Description
    :param steering_angles: Description
    """
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(images, steering_angles):
        if np.random.rand() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
            random_bright = .25 + np.random.uniform()
            hsv[:, :, 2] = hsv[:, :, 2] * random_bright
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
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


def prepare_data():
    df = load_driving_log()
    images, steering_angles = preprocess_data(df)
    aug_images, aug_steering_angles = augment_data(images, steering_angles)
    
    return np.array(aug_images), np.array(aug_steering_angles)
    

    
