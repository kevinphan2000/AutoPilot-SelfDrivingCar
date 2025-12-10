"""Test model predictions on multiple images to check for diversity"""
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd

# Load model
model = load_model('model.h5')
print("Model loaded successfully\n")

# Load CSV to get different steering angles
df = pd.read_csv('driving_log.csv', header=None)
df.columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']

# Test on images with various steering angles
test_cases = [
    (0, 'Zero steering'),
    (df[df['Steering'] < -0.3].iloc[0] if len(df[df['Steering'] < -0.3]) > 0 else None, 'Left turn'),
    (df[df['Steering'] > 0.3].iloc[0] if len(df[df['Steering'] > 0.3]) > 0 else None, 'Right turn'),
    (df.iloc[len(df)//2], 'Middle sample'),
]

print("Testing predictions on different images:")
print("=" * 70)

for idx, (row, desc) in enumerate(test_cases):
    if row is None:
        continue
    
    if isinstance(row, int):
        row = df.iloc[row]
    
    # Get image path
    img_path = row['Center'].split('\\')[-1]
    true_steering = row['Steering']
    
    # Load and preprocess image
    image = cv2.imread(f'images/{img_path}')
    if image is None:
        print(f"Could not load: {img_path}")
        continue
        
    image = image[60:135, :, :]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    image_batch = np.array([image])
    
    # Predict
    predicted = float(model.predict(image_batch, verbose=0))
    
    print(f"{idx+1}. {desc:20s} | True: {true_steering:7.3f} | Predicted: {predicted:7.3f} | Diff: {abs(predicted-true_steering):7.3f}")

print("=" * 70)

# Check if predictions are all similar (model collapse)
predictions = []
for i in range(min(20, len(df))):
    img_path = df.iloc[i]['Center'].split('\\')[-1]
    image = cv2.imread(f'images/{img_path}')
    if image is None:
        continue
    image = image[60:135, :, :]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200, 66))
    image = image.astype(np.float32) / 255.0
    pred = float(model.predict(np.array([image]), verbose=0))
    predictions.append(pred)

predictions = np.array(predictions)
print(f"\nPrediction Statistics (first 20 images):")
print(f"  Mean: {predictions.mean():.4f}")
print(f"  Std:  {predictions.std():.4f}")
print(f"  Min:  {predictions.min():.4f}")
print(f"  Max:  {predictions.max():.4f}")

if predictions.std() < 0.01:
    print("\n⚠️  WARNING: Model has collapsed! All predictions are nearly identical.")
    print("    This suggests the model hasn't learned meaningful patterns.")
else:
    print(f"\n✓ Model shows variation in predictions (std={predictions.std():.4f})")
