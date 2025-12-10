"""Test model on ALL images to see full prediction distribution"""
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

model = load_model('model.h5')
df = pd.read_csv('driving_log.csv', header=None)
df.columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']

predictions = []
true_values = []

print("Testing on all images...")
for idx in range(len(df)):
    img_path = df.iloc[idx]['Center'].split('\\')[-1]
    true_steering = df.iloc[idx]['Steering']
    
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
    true_values.append(true_steering)

predictions = np.array(predictions)
true_values = np.array(true_values)

print("\n" + "="*70)
print("PREDICTION ANALYSIS")
print("="*70)

print(f"\nPredicted values:")
print(f"  Mean:   {predictions.mean():7.4f}")
print(f"  Std:    {predictions.std():7.4f}")
print(f"  Min:    {predictions.min():7.4f}")
print(f"  Max:    {predictions.max():7.4f}")
print(f"  Median: {np.median(predictions):7.4f}")

print(f"\nTrue values:")
print(f"  Mean:   {true_values.mean():7.4f}")
print(f"  Std:    {true_values.std():7.4f}")
print(f"  Min:    {true_values.min():7.4f}")
print(f"  Max:    {true_values.max():7.4f}")
print(f"  Median: {np.median(true_values):7.4f}")

print(f"\nError statistics:")
errors = np.abs(predictions - true_values)
print(f"  Mean Absolute Error: {errors.mean():.4f}")
print(f"  Median Absolute Error: {np.median(errors):.4f}")
print(f"  Max Error: {errors.max():.4f}")

# Distribution analysis
print(f"\nPrediction distribution:")
print(f"  Left  (<-0.1): {np.sum(predictions < -0.1):4d} ({100*np.sum(predictions < -0.1)/len(predictions):.1f}%)")
print(f"  Straight:      {np.sum(np.abs(predictions) <= 0.1):4d} ({100*np.sum(np.abs(predictions) <= 0.1)/len(predictions):.1f}%)")
print(f"  Right (>0.1):  {np.sum(predictions > 0.1):4d} ({100*np.sum(predictions > 0.1)/len(predictions):.1f}%)")

print(f"\nTrue distribution:")
print(f"  Left  (<-0.1): {np.sum(true_values < -0.1):4d} ({100*np.sum(true_values < -0.1)/len(true_values):.1f}%)")
print(f"  Straight:      {np.sum(np.abs(true_values) <= 0.1):4d} ({100*np.sum(np.abs(true_values) <= 0.1)/len(true_values):.1f}%)")
print(f"  Right (>0.1):  {np.sum(true_values > 0.1):4d} ({100*np.sum(true_values > 0.1)/len(true_values):.1f}%)")

# Correlation
correlation = np.corrcoef(predictions, true_values)[0, 1]
print(f"\nCorrelation: {correlation:.4f}")

print("\n" + "="*70)
