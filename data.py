import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Docstring for preprocess_data
    
    :param df: Description
    """
    # Extract image paths and steering angles
    image_paths = df['Center'].values
    steering_angles = df['Steering'].values
    # Load image as numpy array
    for i in range(len(image_paths)):
        # trim path to just the filename
        image_paths[i] = image_paths[i].split('/')[-1]
        # load image in images folder
        image = plt.imread('images/' + image_paths[i])
        image_paths[i] = image
        

    

    return image_paths, steering_angles


if __name__ == "__main__":
    df = load_driving_log()
    plot_steering_histogram(df)
