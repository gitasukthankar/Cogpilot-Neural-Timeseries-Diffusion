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
    Dataset of of the CogPilot Physio data. Normalization is done per (expertise, level) group
            
    Returns:
        Array of shape(batch, channels, time)
        
    - conditional on expert/novince label and difficulty level
    - 14 channels: ECG(3) + EDA(2) + EMG(5) + Resp(1) + ACC(3)
    - 4 second windows: 512 time points at 128 Hz
    """
    def __init__(self, index_json, signal_length=512, target_fs=128.0, split='train', cache_runs=True, preload_all=False):
        super().__init__()
        
        self.signal_length = signal_length
        self.target_fs = target_fs
        self.stats_file = "cogpilot_stats.pkl"
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
            log.info("Calculating stats & indexing lengths from training data...")
            
            #self.channel_mean, self.channel_std, self.run_lengths = self._calculate_global_stats()
            self.stats, self.run_lengths = self._calculate_stats()
            
            # Save stats for test
            with open(self.stats_file, 'wb') as f:
                pickle.dump(self.stats, f)
            log.info(f"Saved stats to {self.stats_file}")
                
            # save run lengths
            with open(self.run_lengths_file, 'wb') as f:
                pickle.dump(self.run_lengths, f)
        
        else:
            # load stats computed from training
            if not os.path.exists(self.stats_file):
                print("Stats file not found")
                
            log.info(f"Loading stats from {self.stats_file}")
            with open(self.stats_file, 'rb') as f:
                self.stats = pickle.load(f)
        
        # build windowed sample index
        self.samples = []
        self._build_sample_index()
        
        if self.preload_all:
            log.info("Preloading all runs into memory...")
            self._preload_all_runs()
            log.info(f"Preloaded {len(self.run_cache)} runs")
        
    def _calculate_stats(self):
        """
        Calculate mean and std for each (expertise, level) combination

        This creates 8 sets of statistics:
        - Novice: levels 1-4 (labels 0-3)
        - Expert: levels 1-4 (labels 4-7)
        
        Returns:
            stats: dict mapping label -> {'mean': tensor, 'std': tensor}
            run_lengths: dict mapping run_idx -> length
        """
        run_lengths = {}
        
        # Group runs by their unified label
        # label = (is_expert * 4) + (level - 1)
        grouped_data = {i: [] for i in range(8)}  # Labels 0-7
        
        log.info(f"Scanning {len(self.run_index)} runs...")
        
        for i, run_info in enumerate(self.run_index):
            # synchronize the run
            df_synch = self.synch.synchronize_run(run_info['files'])
            
            # save run length
            run_lengths[i] = len(df_synch)
            
            # Extract numeric data
            numeric_cols = df_synch.select_dtypes(include=[np.number]).columns
            vals = df_synch[numeric_cols].values  # Shape: (n_samples, 14)
            
            # Calculate unified label
            is_expert = run_info['is_expert']
            level = run_info['level']
            unified_label = (is_expert * 4) + (level - 1)
            
            # Add to appropriate group
            grouped_data[unified_label].append(vals)
            
            # Debug first run
            if i == 0:
                log.info(f"\nDEBUG - First run:")
                log.info(f"  Subject: {run_info['subject']}, Run: {run_info['run_id']}")
                log.info(f"  Expertise: {'Expert' if is_expert else 'Novice'}, Level: {level}")
                log.info(f"  Unified label: {unified_label}")
                log.info(f"  Shape: {vals.shape}")
                log.info(f"  ECG_LL_RA - Mean: {vals[:, 0].mean():.2f}, Std: {vals[:, 0].std():.2f}, Range: [{vals[:, 0].min():.2f}, {vals[:, 0].max():.2f}]")
        
        channel_names = [
            'ECG_LL_RA', 'ECG_LA_RA', 'ECG_VX_RL',
            'PPG', 'EDA',
            'ACC_FA_X', 'ACC_FA_Y', 'ACC_FA_Z', 'EMG_FLEX', 'EMG_EXT',
            'RESP',
            'ACC_T_X', 'ACC_T_Y', 'ACC_T_Z'
        ]
        
        stats = {}
        
        for label in range(8):
            if not grouped_data[label]:
                log.warning(f"Label {label}: No data found!")
                continue
            
            # Concatenate all runs for this label
            combined = np.concatenate(grouped_data[label], axis=0)
            
            # Compute mean and std
            mean_val = np.mean(combined, axis=0)
            std_val = np.std(combined, axis=0)
            std_val = np.maximum(std_val, 1e-8)  # Prevent division by zero
            
            # Convert to torch tensors with shape (14, 1) for broadcasting
            mean_tensor = torch.from_numpy(mean_val).float().unsqueeze(1)
            std_tensor = torch.from_numpy(std_val).float().unsqueeze(1)
            
            stats[label] = {
                'mean': mean_tensor,
                'std': std_tensor
            }
            
            # Log statistics
            is_expert = label // 4
            level = (label % 4) + 1
            expertise = "Expert" if is_expert else "Novice"
            n_runs = len(grouped_data[label])
            n_samples = combined.shape[0]
            
            print(f"\nLabel {label}: {expertise}, Level {level}")
            print(f"  Runs: {n_runs}, Total samples: {n_samples}")
            print(f"  Per-channel statistics:")
            
            for idx, name in enumerate(channel_names):
                print(f"    {name:12s}: mean = {mean_val[idx]:8.3f}, std = {std_val[idx]:8.3f}")
        
        return stats, run_lengths
    
    def _compute_run_lengths(self):
        """Compute run lengths for test split"""
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
        df_sync = self.synch.synchronize_run(run_info['files'])
        
        # Extract numeric data
        numeric_cols = df_sync.select_dtypes(include=[np.number]).columns
        data = df_sync[numeric_cols].values  # (n_samples, 14)
        
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
        label_val = sample_info['label']
        
        # Load run
        data = self._load_run(run_idx)
        
        # Extract window
        start = sample_info['start_idx'] * self.stride
        end = start + self.signal_length
        
        # Get tensor slice: (Channels, Time) = (14, 512)
        x = data[:, start:end]
        
        # Pad if needed
        if x.shape[1] < self.signal_length:
            pad_length = self.signal_length - x.shape[1]
            x = torch.nn.functional.pad(x, (0, pad_length), mode='replicate')
            
        # BEFORE normalization
        if indx == 0:  # Print once
            print(f"\n=== DATASET DEBUG (index={indx}) ===")
            print(f"Label: {label_val}")
            print(f"Raw ECG_LL_RA: mean={x[0].mean():.2f}, std={x[0].std():.2f}, range=[{x[0].min():.2f}, {x[0].max():.2f}]")
        
        # Normalization
        stats = self.stats[label_val]
        x_norm = (x - stats['mean']) / stats['std']
        
        # AFTER normalization
        if indx == 0:
            print(f"Normalized ECG_LL_RA: mean={x_norm[0].mean():.4f}, std={x_norm[0].std():.4f}")
            print(f"Expected: mean≈0.0, std≈1.0")
            print(f"=================================\n")

        label_tensor = torch.tensor(label_val, dtype=torch.long) # <- new change, unified label
        # cond_emb expects (batch, cond_dim, length)

        cond_tensor = torch.full((1, self.signal_length), float(label_val), dtype=torch.float32)
        # Shape: (1, 512) → becomes (batch, 1, 512) after DataLoader batching
        
        return {
            "signal": x_norm,
            "cond": cond_tensor,  # (1,) → becomes (batch, 1) after batching, new thing: # (1, 512) → becomes (batch, 1, 512)
            "label": label_tensor # only used for expierments
        }
       
            