"""Inspect the trained model architecture and weights"""
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model.h5')

print("MODEL ARCHITECTURE:")
print("=" * 60)
model.summary()

print("\n\nWEIGHT STATISTICS:")
print("=" * 60)
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) > 0:
        weight_matrix = weights[0]
        bias = weights[1] if len(weights) > 1 else None
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Weight shape: {weight_matrix.shape}")
        print(f"  Weight mean: {weight_matrix.mean():.6f}")
        print(f"  Weight std:  {weight_matrix.std():.6f}")
        print(f"  Weight min:  {weight_matrix.min():.6f}")
        print(f"  Weight max:  {weight_matrix.max():.6f}")
        if bias is not None:
            print(f"  Bias mean:   {bias.mean():.6f}")
            print(f"  Bias std:    {bias.std():.6f}")

# Check final layer specifically
print("\n\nFINAL LAYER WEIGHTS (critical for output):")
print("=" * 60)
final_weights = model.layers[-1].get_weights()
print(f"Weight: {final_weights[0].flatten()[:20]}")  # First 20 weights
print(f"Bias: {final_weights[1]}")
