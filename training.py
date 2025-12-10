import numpy as np

# batching training data
def data_generator(images, steering_angles, batch_size=32, shuffle=True):
    """
    Function to generate batches of training data
    
    :param images: images
    :param steering_angles: steering angles
    :param batch_size: batch size
    :param shuffle: whether to shuffle data each epoch
    :return: batches of (X_train, y_train)
    """
    num_samples = len(images)
    while True:  # Loop forever so the generator never terminates
        if shuffle:
            indices = np.random.permutation(num_samples)
            images = images[indices]
            steering_angles = steering_angles[indices]
        
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
    train_generator = data_generator(train_images, train_steering_angles, batch_size, shuffle=True)
    validation_generator = data_generator(validation_images, validation_steering_angles, batch_size, shuffle=False)
    
    # Add callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Train the model using fit_generator
    history = model.fit(train_generator,
                       steps_per_epoch=max(1, len(train_images) // batch_size),
                       validation_data=validation_generator,
                       validation_steps=max(1, len(validation_images) // batch_size),
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=1)
    return model



# creating a model
def create_model():
    """
    Function to create the CNN model based on NVIDIA architecture
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Lambda, MaxPooling2D

    model = Sequential()
    # Convolutional layers (images are pre-normalized in preprocessing)
    # Convolutional Layer 1 (5,5) kernel with 24 filters and stride of 2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
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
    # Fully connected layers with reduced dropout
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.1))
    # Output layer
    model.add(Dense(1))
    
    # Compile with lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])
    
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
    
    # Better balance the dataset - reduce straight driving bias
    # Keep all turns, but only 10% of straight segments
    non_straight_indices = np.where(np.abs(train_steering_angles) > 0.15)[0]
    straight_indices = np.where(np.abs(train_steering_angles) <= 0.15)[0]
    
    # Keep only 10% of straight samples to force learning turns
    num_straight_to_keep = max(len(non_straight_indices) // 2, int(len(straight_indices) * 0.1))
    reduced_straight_indices = np.random.choice(straight_indices, size=num_straight_to_keep, replace=False)
    balanced_indices = np.concatenate((non_straight_indices, reduced_straight_indices))
    
    # Shuffle to mix straight and turning samples
    np.random.shuffle(balanced_indices)

    # Apply balanced indices to training data
    train_images = np.array(list(np.array(train_images)[balanced_indices]))
    train_steering_angles = np.array(list(np.array(train_steering_angles)[balanced_indices]))

    # Create model
    model = create_model()
    
    # Print model summary
    model.summary()
    print(f"\nTraining samples: {len(train_images)}")
    print(f"Validation samples: {len(validation_images)}")
    print(f"Steering angle range: [{train_steering_angles.min():.2f}, {train_steering_angles.max():.2f}]")
    print(f"Mean steering angle: {train_steering_angles.mean():.4f}")
    print(f"Std steering angle: {train_steering_angles.std():.4f}")
    
    # Check distribution
    left_turns = np.sum(train_steering_angles < -0.15)
    right_turns = np.sum(train_steering_angles > 0.15)
    straight = np.sum(np.abs(train_steering_angles) <= 0.15)
    print(f"\nDistribution after balancing:")
    print(f"  Left turns:   {left_turns:4d} ({100*left_turns/len(train_steering_angles):.1f}%)")
    print(f"  Straight:     {straight:4d} ({100*straight/len(train_steering_angles):.1f}%)")
    print(f"  Right turns:  {right_turns:4d} ({100*right_turns/len(train_steering_angles):.1f}%)")
    
    # Train model with smaller batch size for better gradient updates
    print("\nStarting training...")
    model = train_model(model, train_images, train_steering_angles,
                        validation_images, validation_steering_angles, 
                        batch_size=16, epochs=30)
    
    # Save the trained model
    model.save('model.h5')
    print("\nModel saved to model.h5")
    print("Best model saved to best_model.h5")