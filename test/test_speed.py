"""
Diagnostic script to identify training bottlenecks
Run this to profile your training loop
"""
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import logging
import sys
import os

# Add project root to path if running from a subfolder
sys.path.append(".") 

from CogPilot.dataset import CogPilotDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def profile_dataset_loading():
    """Profile how long dataset operations take"""
    
    log.info("=" * 60)
    log.info("PROFILING DATASET LOADING")
    log.info("=" * 60)
    
    # Initialize dataset
    log.info("Creating dataset...")
    start = time.time()
    dataset = CogPilotDataset(
        index_json="data/processed/dataset_index.json",
        signal_length=512,
        target_fs=128.0,
        split='train',
        cache_runs=True, 
        preload_all=True
    )
    init_time = time.time() - start
    log.info(f"Dataset initialization: {init_time:.2f}s")
    log.info(f"Total samples: {len(dataset)}")
    
    # Test single item access
    log.info("\nTesting single item access (first time)...")
    start = time.time()
    item = dataset[0]
    first_access = time.time() - start
    log.info(f"First access: {first_access:.3f}s")
    
    # Test cached access
    log.info("Testing cached access (should be faster)...")
    start = time.time()
    item = dataset[0]
    cached_access = time.time() - start
    log.info(f"Cached access: {cached_access:.3f}s")
    
    if cached_access > 0.01:
        log.warning("⚠️  Cached access is slow! Caching may not be working.")
    
    # Test accessing multiple samples
    log.info("\nTesting 100 random samples...")
    indices = np.random.randint(0, len(dataset), 100)
    start = time.time()
    for idx in indices:
        _ = dataset[idx]
    batch_time = time.time() - start
    log.info(f"100 samples: {batch_time:.2f}s ({batch_time/100*1000:.1f}ms per sample)")
    
    return dataset


def profile_dataloader(dataset, num_workers=4):
    """Profile DataLoader with different configurations"""
    
    log.info("\n" + "=" * 60)
    log.info("PROFILING DATALOADER")
    log.info("=" * 60)
    
    batch_size = 32
    
    # Test different worker counts
    for workers in [0, 2, 4, 8]:
        log.info(f"\nTesting with {workers} workers...")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True if workers > 0 else False
        )
        
        # Time one full epoch
        start = time.time()
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count >= 10:  # Test first 10 batches
                break
        
        elapsed = time.time() - start
        samples_loaded = batch_count * batch_size
        
        log.info(f"  {workers} workers: {elapsed:.2f}s for {samples_loaded} samples")
        log.info(f"  Throughput: {samples_loaded/elapsed:.1f} samples/sec")
        log.info(f"  Per batch: {elapsed/batch_count*1000:.1f}ms")


def profile_training_step(dataset):
    """Profile a single training step"""
    
    log.info("\n" + "=" * 60)
    log.info("PROFILING TRAINING STEP")
    log.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Create a simple model for testing
    from ntd.networks import AdaConv
    network = AdaConv(
        signal_length=512,
        signal_channel=14,
        cond_dim=1,
        hidden_channel=64,
        in_kernel_size=3,
        out_kernel_size=3,
        slconv_kernel_size=3,
        num_scales=4,
        num_blocks=2,
        num_off_diag=1,
        use_pos_emb=True,
        padding_mode='circular',
        use_fft_conv=False,
    ).to(device)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Profile forward pass
    log.info("\nProfiling forward pass...")
    batch = next(iter(loader))
    signal = batch['signal'].to(device)
    cond = batch['cond'].to(device)
    
    # Warm up
    _ = network(signal, torch.randn_like(signal), torch.ones(32, 1).to(device), cond)
    
    # Time forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(10):
        output = network(signal, torch.randn_like(signal), torch.ones(32, 1).to(device), cond)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.time() - start) / 10
    
    log.info(f"Forward pass: {forward_time*1000:.1f}ms")
    
    # Profile backward pass
    log.info("Profiling backward pass...")
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        output = network(signal, torch.randn_like(signal), torch.ones(32, 1).to(device), cond)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    backward_time = (time.time() - start) / 10
    
    log.info(f"Forward + Backward + Optimizer: {backward_time*1000:.1f}ms")
    
    return forward_time, backward_time


def estimate_epoch_time(dataset, forward_time, backward_time, num_workers=4):
    """Estimate total epoch time"""
    
    log.info("\n" + "=" * 60)
    log.info("EPOCH TIME ESTIMATION")
    log.info("=" * 60)
    
    batch_size = 32
    num_batches = len(dataset) // batch_size
    
    # Component times
    data_load_time = 0.01  # Assume 10ms per batch if cached
    train_step_time = backward_time
    
    total_time = (data_load_time + train_step_time) * num_batches
    
    log.info(f"Dataset size: {len(dataset)} samples")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Batches per epoch: {num_batches}")
    log.info(f"")
    log.info(f"Estimated per-batch times:")
    log.info(f"  Data loading: {data_load_time*1000:.1f}ms")
    log.info(f"  Training step: {train_step_time*1000:.1f}ms")
    log.info(f"  Total: {(data_load_time + train_step_time)*1000:.1f}ms")
    log.info(f"")
    log.info(f"Estimated epoch time: {total_time/60:.1f} minutes")
    log.info(f"Estimated 150 epochs: {total_time*150/3600:.1f} hours")


def main():
    """Run all profiling"""
    
    log.info("Starting training speed diagnostic...\n")
    
    try:
        # Profile dataset
        dataset = profile_dataset_loading()
        
        # Profile DataLoader
        profile_dataloader(dataset)
        
        # Profile training step
        forward_time, backward_time = profile_training_step(dataset)
        
        # Estimate total time
        estimate_epoch_time(dataset, forward_time, backward_time)
        
        log.info("\n" + "=" * 60)
        log.info("DIAGNOSTIC COMPLETE")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()