# load model and make prediction
def testing():
    from tensorflow.keras.models import load_model
    import numpy as np
    import cv2

    # Load the trained model
    model = load_model('model.h5')

    # Load a test image
    image = cv2.imread('images/center_2025_12_06_11_10_55_416.jpg')
    image = image[60:135, :, :]  # Crop the image
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0  # Normalize the image
    image = np.array([image])  # Add batch dimension

    # Make prediction
    steering_angle = float(model.predict(image))
    print(f'Predicted steering angle: {steering_angle}')




if __name__ == "__main__":
    testing()