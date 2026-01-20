#!/usr/bin/env python3
"""
Diagnostic script to identify training issues.
"""

import numpy as np
import torch
import torch.nn as nn
from data_loader import DASDataLoader, fft

# Load a small sample of data
decim_dict = {'regular': 50}
parser_loader = DASDataLoader(
    data_dir='/nobackup/carda/datasets/DAS-dataset/data',
    sample_len=2048,
    transform=fft,
    fsize=8192,
    shift=2048,
    decimate=decim_dict,
)

x, y = parser_loader.parse_dataset()

print("=" * 70)
print("DATA ANALYSIS")
print("=" * 70)
print(f"Data shape: {x.shape}")
print(f"Data stats:")
print(f"  Min: {x.min():.4f}")
print(f"  Max: {x.max():.4f}")
print(f"  Mean: {x.mean():.4f}")
print(f"  Std: {x.std():.4f}")
print(f"  Contains NaN: {np.isnan(x).any()}")
print(f"  Contains Inf: {np.isinf(x).any()}")

# Check class distribution
y_classes = np.argmax(y, axis=1)
unique, counts = np.unique(y_classes, return_counts=True)
print(f"\nClass distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls} ({parser_loader.encoder.classes_[cls]}): {count} samples ({100*count/len(y_classes):.2f}%)")

print("\n" + "=" * 70)
print("MODEL ANALYSIS")
print("=" * 70)

# Test model with actual data
class DASEventClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=9):
        super(DASEventClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=0)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=7, stride=1, padding=0)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 126, 1024)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x_sig = self.sigmoid(x)
        x = self.fc2(x_sig)
        return x, x_sig

model = DASEventClassifier()
sample_data = torch.FloatTensor(x[:10])

with torch.no_grad():
    output, sigmoid_output = model(sample_data)
    
print(f"Input stats: min={sample_data.min():.4f}, max={sample_data.max():.4f}, mean={sample_data.mean():.4f}, std={sample_data.std():.4f}")
print(f"\nAfter sigmoid stats:")
print(f"  Min: {sigmoid_output.min():.4f}")
print(f"  Max: {sigmoid_output.max():.4f}")
print(f"  Mean: {sigmoid_output.mean():.4f}")
print(f"  Std: {sigmoid_output.std():.4f}")
print(f"  Values near 0: {(sigmoid_output < 0.1).sum().item()}")
print(f"  Values near 1: {(sigmoid_output > 0.9).sum().item()}")
print(f"\nOutput logits stats:")
print(f"  Min: {output.min():.4f}")
print(f"  Max: {output.max():.4f}")
print(f"  Mean: {output.mean():.4f}")
print(f"  Std: {output.std():.4f}")

probs = torch.softmax(output, dim=1)
print(f"\nSoftmax probabilities (first sample):")
print(probs[0].numpy())
print(f"  Max prob: {probs[0].max():.4f}")
print(f"  Entropy: {-(probs[0] * torch.log(probs[0] + 1e-10)).sum():.4f}")

# Test gradient flow
sample_data.requires_grad = True
output, _ = model(sample_data)
loss = nn.CrossEntropyLoss()(output, torch.LongTensor([y_classes[0]] * 10))
loss.backward()

print(f"\nGradient analysis:")
print(f"  Input grad norm: {sample_data.grad.norm():.4f}")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad_norm={param.grad.norm():.4f}, grad_mean={param.grad.mean():.4f}")
    else:
        print(f"  {name}: NO GRADIENT")



