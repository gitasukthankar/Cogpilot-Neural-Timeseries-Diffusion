import sys
import os
import torch
import numpy as np
import pprint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import random

# Add project root to path if running from a subfolder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CogPilot.dataset import CogPilotDataset

CONFIG = {
    "index_json": "data/processed/dataset_index.json",
    "signal_length": 512,
    "target_fs": 128.0,
    "cache_runs": True, 
    "preload_all": True
}

def decode_label(label_index):
    """Helper to decode the new 0-7 label back to human readable text."""
    # Formula: label = (is_expert * 4) + (difficulty - 1)
    is_expert = label_index // 4
    difficulty = (label_index % 4) + 1
    
    expert_str = "Expert" if is_expert else "Novice"
    return f"{expert_str} - Level {difficulty}"

# -----------------------------
# Initialize & Verify Split Logic
# -----------------------------

print("\n===== VERIFYING SPLITS =====")
# Initialize both datasets
print("Loading Training Set...")
train_ds = CogPilotDataset(split="train", **CONFIG)
print("\nLoading Testing Set...")
test_ds = CogPilotDataset(split="test", **CONFIG)

# Extract Run IDs from the internal index
# run_index is a list of dicts: [{'run_id': '...', ...}, ...]
train_ids = set([f"{r['subject']}_{r['run_id']}" for r in train_ds.run_index])
test_ids  = set([f"{r['subject']}_{r['run_id']}" for r in test_ds.run_index])

n_total = len(train_ids) + len(test_ids)
print(f"Train Windows: {len(train_ds)}")
print(f"Test Windows: {len(test_ds)}")
print(f"Train Runs: {len(train_ids)}")
print(f"Test Runs:  {len(test_ids)}")
print(f"Total Runs:        {n_total}")
print(f"Avg Windows/Run (train): {len(train_ids)/n_total:.1%} / {len(test_ids)/n_total:.1%}")

# Check for no overlap in runs
intersection = train_ids.intersection(test_ids)

if len(intersection) == 0:
    print("\n✅ SUCCESS: No overlap between Training and Testing sets.")
else:
    print(f"\n❌ CRITICAL ERROR: Found {len(intersection)} recordings in BOTH sets!")
    print(f"Overlapping: {list(intersection)}...")
    
train_subjs = set([x.split('_')[0] for x in train_ids])
test_subjs = set([x.split('_')[0] for x in test_ids])

subj_overlap = train_subjs.intersection(test_subjs)

print("\n--- Subject Leakage Check ---")
if len(subj_overlap) == 0:
    print("✅ SUCCESS: Subjects are strictly separated.")
    print(f"Train Subjects: {len(train_subjs)}")
    print(f"Test Subjects:  {len(test_subjs)}")
else:
    print(f"⚠️ WARNING: {len(subj_overlap)} subjects appear in both sets (Split is by Run, not Subject).")
   
   
""" 
# check the global mean and std
means = dataset.channel_mean.squeeze()
stds = dataset.channel_std.squeeze()

print(f"Ch 0 (ECG) Mean: {means[0]:.4f} | Std: {stds[0]:.4f}")
print(f"Ch 1 (ECG) Mean: {means[1]:.4f} | Std: {stds[1]:.4f}")
print(f"Ch 2 (ECG) Mean: {means[2]:.4f} | Std: {stds[2]:.4f}  <-- Should be ~338")
print(f"Ch 3 (EDA) Mean: {means[3]:.4f} | Std: {stds[3]:.4f}")

if means[2] > 100:
    print("\n✅ SUCCESS: Global Mean captures the physical offset.")
else:
    print("\n❌ FAILURE: Global Mean is too small. Did normalization fail?")"""


# -----------------------------
# Test __getitem__ & label logic
# -----------------------------
print("\n===== GETITEM TEST =====")
sample_index = 32241
sample = train_ds[sample_index] # grab a random sample

# Extract components
signal = sample["signal"]
cond = sample["cond"]
label = sample["label"]

print(f"Signal shape: {signal.shape}") # Expect: [14, 512]
#print(f"Cond shape:   {cond.shape}")   # Expect: [1, 512] <- old, 
print(f"Label:        {label.item()}") # Expect: (0-7)
print(f"Decoded:      {decode_label(label.item())}")

# Verify tensor shapes
if signal.shape == (14, 512):
    print("SUCCESS: Signal shape is correct for AdaConv (Channels, Time).")
else:
    print(f"ERROR: Signal shape mismatch! Got {signal.shape}")

if label.dim() == 0:
    print("SUCCESS: Label is a scalar (correct for nn.Embedding).")
else:
    print(f"ERROR: Label should be scalar, got shape {label.shape}")

# -----------------------------
# Test Cache
# -----------------------------
print("\n===== CACHING PERFORMANCE TEST =====")
# Access a sample from a new run (uncached)
start = time.time()
_ = train_ds 
print(f"First access (Load + Synch): {time.time() - start:.5f}s")

# Access it again (should be faster)
start = time.time()
_ = train_ds
print(f"Second access (Cached): {time.time() - start:.5f}s")

# -----------------------------
# Test DataLoader Batch
# -----------------------------
print("\n===== DATALOADER TEST =====")
loader = DataLoader(train_ds, batch_size=32, shuffle=True)

batch = next(iter(loader))

print(f"Signal shape: {batch['signal'].shape}")  # Should be (32, 14, 512)
print(f"Cond shape:   {batch['cond'].shape}")    # Should be (32, 1)
print(f"Label shape:  {batch['label'].shape}")   # Should be (32,)

print("\nCond values (first 5):", batch['cond'][:5].squeeze().tolist())
print("Label values (first 5):", batch['label'][:5].tolist())

""" # Verify they match
assert torch.allclose(batch['cond'].squeeze(), batch['label'].float()), "Cond and label should match!"
print("✅ Cond and label values match!") """

print(f"Batch Label Shape: {batch['label'].shape}")      # Expect: ['signal', 'cond', 'label']
print("Sample Labels in Batch:", batch['label'].tolist())
for lbl in batch['label'].tolist():
    print(f"  - {decode_label(lbl)}")

# -----------------------------
#  Plot a raw window
# -----------------------------
""" plot_example = True

if plot_example:
    plt.figure(figsize=(12, 5))
    plt.title(f"Example Window (Channel 0) - Label: {label.item()}")
    
    # Use the signal we extracted earlier
    plt.plot(signal[0].numpy()) 
    
    plt.xlabel("Time points")
    plt.ylabel("Amplitude (Standardized)")
    plt.grid(True, alpha=0.3)
    plt.show()
 """