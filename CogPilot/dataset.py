import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
from ntd.utils.utils import standardize_array
from pathlib import Path
from .preprocessing import CogPilotPhysiologicalDataSynchronizer
import pickle
import json
import logging
from functools import lru_cache
log = logging.getLogger(__name__)

class CogPilotDataset(Dataset):
    """
    Dataset of of the CogPilot Physio data
            
    Returns:
        Array of shape(batch, channels, time)
        
    - conditional on expert/novince label
    - 14 channels: ECG(3) + EDA(2) + EMG(5) + Resp(1) + ACC(3)
    - 4 second windows: 512 time points at 128 Hz
    """
    def __init__(self, index_json, signal_length=512, target_fs=128.0, split='train', cache_runs=True, preload_all=False):
        super().__init__()
        
        self.signal_length = signal_length
        self.target_fs = target_fs
        self.stats_file = "cogpilot_global_stats.pkl"
        self.run_lengths_file = "data/processed/cogpilot_run_lengths.pkl"
        self.stride = self.signal_length // 2
        
        self.cache_runs = cache_runs
        self.preload_all = preload_all
            
        self.run_index = self._load_and_split_index(index_json, split)
        
        # Synchronizer
        self.synch = CogPilotPhysiologicalDataSynchronizer(target_fs=self.target_fs)
        
        self.run_lengths = {}
        
        # Run cache (if enabled)
        if self.cache_runs:
            self.run_cache = {}
            log.info(f"Run caching enabled (cache_runs=True)")
            
        # compute normalization stats
        if split == 'train':
            print("Calculating stats & indexing lengths from training data...")
            
            self.channel_mean, self.channel_std, self.run_lengths = self._calculate_global_stats()
            
            # Save stats for test
            with open(self.stats_file, 'wb') as f:
                pickle.dump({'mean': self.channel_mean, 'std': self.channel_std}, f)
                
            # save run lengths
            with open(self.run_lengths_file, 'wb') as f:
                pickle.dump(self.run_lengths, f)
            
        else:
            # load stats computed from training
            if not os.path.exists(self.stats_file):
                print("Stats file not found")
                
            print(f"Loading global stats from {self.stats_file}")
            with open(self.stats_file, 'rb') as f:
                stats = pickle.load(f)
                
            self.channel_mean = stats['mean']
            self.channel_std = stats['std']
        
        # build windowed sample index
        self.samples = []
        self._build_sample_index()
        
        if self.preload_all:
            log.info("Preloading all runs into memory...")
            self._preload_all_runs()
            log.info(f"Preloaded {len(self.run_cache)} runs")
        
    def _calculate_global_stats(self):
        #Calculate the mean and std, saves length of each run to self.run_lengths  
        #Returns
           # - mean & std
        run_lengths = {}
        sum_x = np.zeros(14)
        sum_sq_x = np.zeros(14)
        total_count = 0
        
        n = 0
        M = np.zeros(14)  # Running mean
        S = np.zeros(14)  # Running sum of squared differences from mean
        
        print(f"Scanning {len(self.run_index)} runs...")
        
        for i, run_info in enumerate(self.run_index):
            # synchronize the run
            #try:
            df_synch = self.synch.synchronize_run(run_info['files'])
            
            # save run length
            run_lengths[i] = len(df_synch)
            
            numeric_cols = df_synch.select_dtypes(include=[np.number]).columns
            vals = df_synch[numeric_cols].values  # Shape: (n_samples, 14)
            
            
            # print stuff
            if i == 0:
                print(f"\n DEBUG")
                print(f"Run info: {run_info['subject']}, run {run_info['run_id']}")
                print(f"Synch Shape: {vals.shape}")
                print(f"Column 0 (ECG LL RA)") # I think
                print(f"    Mean: {vals[:,  0].mean():.2f}")
                print(f"    Std: {vals[:,  0].std():.2f}")
                print(f"    Range: [{vals[:, 0].min():.2f}, {vals[:, 0].max():.2f}]")
                print(f"\nColumn 2 (ECG LA RA):")
                print(f"  Mean: {vals[:, 1].mean():.2f}")
                print(f"  Std: {vals[:, 1].std():.2f}")
                print(f"  Range: [{vals[:, 1].min():.2f}, {vals[:, 2].max():.2f}]")
                print(f"\nColumn 2 (ECG_VX_RL):")
                print(f"  Mean: {vals[:, 2].mean():.2f}")
                print(f"  Std: {vals[:, 2].std():.2f}")
                print(f"  Range: [{vals[:, 2].min():.2f}, {vals[:, 2].max():.2f}]")
            
            # Welford's algorithm: update mean and variance incrementally
            for sample in vals:
                n += 1
                delta = sample - M
                M += delta / n
                delta2 = sample - M
                S += delta * delta2          
        
        # Compute final mean and std
        mean_val = M
        variance_val = S / (n - 1) if n > 1 else np.zeros(14)
        std_val = np.sqrt(variance_val)
        
        # Add small epsilon to prevent division by zero
        std_val = np.maximum(std_val, 1e-8)
        
        # Print per-channel stats
        channel_names = [
            'ECG_LL_RA', 'ECG_LA_RA', 'ECG_VX_RL',
            'PPG', 'EDA',
            'ACC_FA_X', 'ACC_FA_Y', 'ACC_FA_Z', 'EMG_FLEX', 'EMG_EXT',
            'RESP',
            'ACC_T_X', 'ACC_T_Y', 'ACC_T_Z'
        ]
        print(f"\nPer-channel statistics:")
        for idx, name in enumerate(channel_names):
            print(f"  {name:12s}: mean = {mean_val[idx]:8.3f}, std = {std_val[idx]:8.3f}")
        
        # Convert to torch tensors with shape (14, 1) for broadcasting
        mean_tensor = torch.from_numpy(mean_val).float().unsqueeze(1)
        std_tensor = torch.from_numpy(std_val).float().unsqueeze(1)
        
        return mean_tensor, std_tensor, run_lengths
    
    def _compute_run_lengths(self):
        """Compute run lengths for test split."""
        run_lengths = {}
        for i, run_info in enumerate(self.run_index):
            try:
                df_sync = self.synch.synchronize_run(run_info['files'])
                run_lengths[i] = len(df_sync)
            except Exception as e:
                log.warning(f"  Skipping run {i}: {e}")
        return run_lengths
            
    def _load_and_split_index(self, index_json, split):
        # Load index
        with open(index_json, "r") as f:
            index_json = json.load(f)
            
        # split data by runs
        n_train = int(len(index_json) * 0.8)
       
        if split == 'train': # First 80%
            return index_json[:n_train]
        else: # test (Remaining 20%)
            return index_json[n_train:]
        
    def _build_sample_index(self):
        """
        Build index of all windows with 50% overlap
        Each entry in self.samples represents one 4-second window
        """
        
        n_samples = 0
        for run_indx, run_info in enumerate(self.run_index):
            
            if run_indx in self.run_lengths:
                n_samples = self.run_lengths[run_indx]
            else:
                # We are in test set
                df_sync = self.synch.synchronize_run(run_info['files'])
                n_samples = len(df_sync)
            
            # Calculate number of windows for this run
            n_windows = (n_samples - self.signal_length) // self.stride + 1
            
            is_expert = run_info['is_expert']
            difficulty = run_info['level']
            
            # Create unified 8-class label
            # 0-3: Novice Levels 1-4
            # 4-7: Expert Levels 1-4
            # Formula: (Expert * 4) + (Difficulty - 1)
            unified_label = (is_expert * 4) + (difficulty - 1)
            
            # Create entry for each window
            for window_idx in range(n_windows):
                start_idx = window_idx
                
                self.samples.append({
                    'run_idx': run_indx,
                    'start_idx': start_idx, #* self.stride,
                    'is_expert': is_expert,
                    'subject': run_info['subject'],
                    'subject_num': run_info['subject_num'],
                    'run_id': run_info['run_id'],
                    'label': unified_label,
                })
    
    def _preload_all_runs(self):
        """Preload all runs into memory """
        unique_runs = set(s['run_idx'] for s in self.samples)
        for run_idx in unique_runs:
            self._load_run(run_idx)
                
    def _load_run(self, run_idx):
        """Load and cache a single run"""
        
        if run_idx in self.run_cache:
            return self.run_cache[run_idx]
    
        run_info = self.run_index[run_idx]
        
        # DEBUG: Print before synchronization
        print(f"\n=== LOADING RUN {run_idx} ===")
        print(f"Files: {run_info['files']}")
        
        df_sync = self.synch.synchronize_run(run_info['files'])
        
        # DEBUG: Check synchronized data
        print(f"Synchronized data shape: {df_sync.shape}")
        print(f"Columns: {df_sync.columns.tolist()}")
        
        # Extract numeric data
        numeric_cols = df_sync.select_dtypes(include=[np.number]).columns
        data = df_sync[numeric_cols].values  # (n_samples, 14)
        
        # DEBUG: Check extracted data
        print(f"Extracted numeric data shape: {data.shape}")
        print(f"ECG_LL_RA column (assumed first): mean={data[:, 0].mean():.2f}, std={data[:, 0].std():.2f}")
        print(f"Expected: mean around -38, std around 10")
        print("="*40)
        
        # Convert to tensor and cache
        data_tensor = torch.from_numpy(data.T).float()  # (14, n_samples)
        
        if self.cache_runs:
            self.run_cache[run_idx] = data_tensor
        
        return data_tensor
        
        """ if run_idx in self.run_cache:
            return self.run_cache[run_idx]
    
        run_info = self.run_index[run_idx]
        df_sync = self.synch.synchronize_run(run_info['files'])
        
        # Extract numeric data
        numeric_cols = df_sync.select_dtypes(include=[np.number]).columns
        data = df_sync[numeric_cols].values  # (n_samples, 14)
        
        # Convert to tensor and cache
        data_tensor = torch.from_numpy(data.T).float()  # (14, n_samples)
        
        if self.cache_runs:
            self.run_cache[run_idx] = data_tensor
        
        return data_tensor """
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, indx):
        """
        Get one window of data
        
        Returns:
        
            - signal: (channels, time) = (14, 512) tensor
            - cond: (1, 512) dict with expert/novince label
            - label: (1,) tensor for classification experiments
            
            updates
            - label: scalar tensor (0-7)
            - cond: scalar tensor (0-7) same as above
        """
        sample_info = self.samples[indx]
        run_idx = sample_info['run_idx']
        
        # Load run
        data = self._load_run(run_idx)
        
        # Extract window
        start = sample_info['start_idx'] * self.stride
        end = start + self.signal_length
        
        # Get numpy array: (time, channels) - (512, 14)
        # Get tensor slice: (Channels, Time) = (14, 512)
        x = data[:, start:end]  # 
        
        # Ensure correct length
        """ if len(x) < self.signal_length:
            pad_length = self.signal_length - len(x)
            x = np.pad(x, ((0, pad_length), (0, 0)), mode='edge') """
        # Pad if needed
        if x.shape[1] < self.signal_length:
            pad_length = self.signal_length - x.shape[1]
            x = torch.nn.functional.pad(x, (0, pad_length), mode='replicate')
            
        # BEFORE normalization
        if indx == 0:  # Print once
            print(f"\n=== DATASET DEBUG (index={indx}) ===")
            print(f"Raw signal ECG_LL_RA: mean={x[0].mean():.2f}, std={x[0].std():.2f}, range=[{x[0].min():.2f}, {x[0].max():.2f}]")
            print(f"channel_mean shape: {self.channel_mean.shape}, values: {self.channel_mean.flatten()[:3]}")
            print(f"channel_std shape: {self.channel_std.shape}, values: {self.channel_std.flatten()[:3]}")
        
        # Global normalization
        x_norm = (x - self.channel_mean) / self.channel_std
        
        # AFTER normalization
        if indx == 0:
            print(f"Normalized ECG_LL_RA: mean={x_norm[0].mean():.4f}, std={x_norm[0].std():.4f}")
            print(f"Expected: mean≈0.0, std≈1.0")
            print(f"=================================\n")
        
        # Conditioning, using the new label
        label_val = sample_info['label']
        
        # Conditioning        
        #cond_tensor = torch.ones(1, self.signal_length) * is_expert # this is the shape (1, 512) <- but two others say it isn't correct as AdaConv expects (1,)

        # new change, as we aren't sure if the condition is being used correctly
        #cond_tensor = torch.tensor([is_expert], dtype=torch.float32)  # shape (1,)
        #is_expert = float(sample_info['is_expert'])
        #label_tensor = torch.tensor([is_expert], dtype=torch.float32)
        label_tensor = torch.tensor(label_val, dtype=torch.long) # <- new change, unified label
        # cond_emb expects (batch, cond_dim, length)
        
        
        # new thign added
        #cond_tensor = torch.tensor([float(label_val)], dtype=torch.float32)  # Shape: (1,)
        #cond_tensor = torch.tensor([float(label_val)], dtype=torch.float32)  # ← Correct shape, 2nd try
        
        cond_tensor = torch.full((1, self.signal_length), float(label_val), dtype=torch.float32)
        # Shape: (1, 512) → becomes (batch, 1, 512) after DataLoader batching
        
        return {
            "signal": x_norm,
            "cond": cond_tensor,  # (1,) → becomes (batch, 1) after batching, new thing: # (1, 512) → becomes (batch, 1, 512)
            "label": label_tensor # only used for expierments
        }
       
            