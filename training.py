import numpy as np

def data_generator(images, steering_angles, batch_size=32):
    """
    Function to generate batches of training data
    
    :param images: images
    :param steering_angles: steering angles
    :param batch_size: batch size
    :return: batches of (X_train, y_train)
    """
    num_samples = len(images)
    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset + batch_size]
            batch_steering_angles = steering_angles[offset:offset + batch_size]
            
            X_train = np.array(batch_images)
            y_train = np.array(batch_steering_angles)
            yield X_train, y_train

# train the model
def train_model(model, train_images, train_steering_angles,
                validation_images, validation_steering_angles,
                batch_size=32, epochs=5):
    """
    Function to train the model using generators
    
    :param model: model to be trained
    :param train_images: training images
    :param train_steering_angles: training steering angles
    :param validation_images: validation images
    :param validation_steering_angles: validation steering angles
    :param batch_size: batch size
    :param epochs: epochs
    :return: trained model
    """
    # Create generators
    train_generator = data_generator(train_images, train_steering_angles, batch_size)
    validation_generator = data_generator(validation_images, validation_steering_angles, batch_size)
    
    # Train the model using fit_generator
    model.fit(train_generator,
              steps_per_epoch=len(train_images) // batch_size,
              validation_data=validation_generator,
              validation_steps=len(validation_images) // batch_size,
              epochs=epochs)
    return model

# creating a model
def create_model():
    """
    Function to create the CNN model based on NVIDIA architecture
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Lambda, MaxPooling2D

    model = Sequential()
    # Normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    # Convolutional layers
    # Convolutional Layer 1 (5,5) kernel with 24 filters and stride of 2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    # Convolutional Layer 2 (5,5) kernel with 36 filters and stride of 2
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    # Convolutional Layer 3 (5,5) kernel with 48 filters and stride of 2
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    # Convolutional Layer 4 (3,3) kernel with 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Convolutional Layer 5 (3,3) kernel with 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Flattening layer
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # Output layer
    model.add(Dense(1))
    
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model


if __name__ == "__main__":
    from data import load_driving_log, preprocess_data, augment_data
    from sklearn.model_selection import train_test_split

    # Load and preprocess data
    df = load_driving_log()
    images, steering_angles = preprocess_data(df)
    
    # Augment data
    images, steering_angles = augment_data(images, steering_angles)
    
    # Split data into training and validation sets
    train_images, validation_images, train_steering_angles, validation_steering_angles = train_test_split(
        images, steering_angles, test_size=0.2, random_state=42)
    
    # Create model
    model = create_model()
    
    # Train model
    model = train_model(model, train_images, train_steering_angles,
                        validation_images, validation_steering_angles,
                        batch_size=32, epochs=5)
    
    # Save the trained model
    model.save('model.h5')