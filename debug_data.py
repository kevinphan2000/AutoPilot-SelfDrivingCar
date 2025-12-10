from data import load_driving_log, preprocess_data
import numpy as np

df = load_driving_log()
print(f"Total samples in CSV: {len(df)}")

images, angles = preprocess_data(df)
print(f"\nAfter preprocessing:")
print(f"Number of images: {len(images)}")
print(f"First image shape: {images[0].shape}")
print(f"First image dtype: {images[0].dtype}")
print(f"First image min: {images[0].min():.3f}, max: {images[0].max():.3f}")
print(f"First steering angle: {angles[0]}")
print(f"\nSteering angle stats:")
print(f"Min: {angles.min():.3f}, Max: {angles.max():.3f}, Mean: {angles.mean():.3f}")
