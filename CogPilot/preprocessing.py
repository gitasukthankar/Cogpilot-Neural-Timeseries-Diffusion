import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from typing import Dict, List, Optional, Tuple
from fractions import Fraction

"""
Preprocessing utilities for CogPilot physiological data
Includes: Label generation, dataset indexing, and synchronization of data
"""
# PART 1: Generate expert labels and assign to all subjects
def generate_expert_labels(root_dir: str, output_path: str = 'subject_labels.json') -> Dict:
    """Generate expert/novice labels from PerfMetrics.csv."""
    perf_metrics_path = None
    
    for root, dirs, files in os.walk(root_dir):
        if "PerfMetrics.csv" in files:
            perf_metrics_path = os.path.join(root, "PerfMetrics.csv")
            break
    
    if not perf_metrics_path:
        raise FileNotFoundError("PerfMetrics.csv not found")
    
    df = pd.read_csv(perf_metrics_path)
    
    if "cumulative_total_error" not in df.columns:
        raise ValueError("Column 'cumulative_total_error' missing")
    
    # Average across difficulty levels
    subject_errors = df.groupby('subject')['cumulative_total_error'].mean()
    median_error = subject_errors.median()
    
    # Below median = expert (1), above = novice (0)
    labels = (subject_errors < median_error).astype(int).to_dict()
    
    #print(f"Median Error: {median_error:.2f}")
    #print(f"Experts: {sum(labels.values())}, Novices: {len(labels) - sum(labels.values())}")
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    return labels


# PART 2: index the data
def index_dataset(data_root, labels_path='data/processed/subject_labels.json', output_json = 'data/processed/dataset_index.json'):
    """
    Traverse dataPackage folder, creates a JSON of metadata for each run and its modalities
    
    Args:
        data_root: Root directory containing subject/session/run data
        output_json: Path to save the index JSON
    
    Returns:
        index: List of run dictionaries with file paths and metadata
    
    Directory structure expected:
        data_root/
        └── task-ils/
            └── sub-cp004/
                └── ses-20210330/
                    └── level-01B_run-001/
                        ├── *_stream-lslshimmerecg_*_dat.csv
                        ├── *_stream-lslshimmerecg_*_hea.csv
                        ├── *_stream-lslshimmereda_*_dat.csv
    """
    run_regex = re.compile(r'run-(\d+)$')
    subject_regex = re.compile(r'cp0(\d+)$')
    level_regex = re.compile(r'level-(\d+)')
  
    index = []
    modalities = {
        'ecg': 'stream-lslshimmerecg',
        'eda': 'stream-lslshimmereda',
        'emg': 'stream-lslshimmeremg',
        'resp': 'stream-lslshimmerresp',
        'acc': 'stream-lslshimmertorsoacc'
    }
    
    stats = {
        'total_runs_found': 0,
        'valid_runs': 0,
        'skipped_runs': 0,
        'missing_modalities': {mod: 0 for mod in modalities.keys()}
    }
    
    #expert_labels_path = 'subject_labels.json'

    # load in json expert file
    with open(labels_path, 'r') as f:
        expert_data = json.load(f)

    # Walk the directory- Structure: sub-XXX / ses-XXX / task-ils
    for root, dirs, files in os.walk(data_root):
        # Sort subdirectory of runs in a subject folder by the run count
        dirs.sort(key=lambda d: int(run_regex.search(d).group(1)) if run_regex.search(d) else float("inf"))

        if not dirs:
            # Identify unique runs. Files that end in..._run-XXX-dat.csv
            # We extract the run identifiers (e.g., '001', '002')
            run_ids = set([f.split('_run-')[-1].split('_')[0] for f in files if '_run-' in f])
            
            for run in run_ids:
                run_files = [f for f in files if f"_run-{run}" in f]
                
                has_all_streams = True
                modality_files = {}
                mismatch_by_modality = {}
                
                # Check if we have all the modality files
                for mod, tag in modalities.items():
                    # Find the file that matches the tags
                    d_file = next((f for f in run_files if tag in f and not "hea" in f), None)
                    h_file = next((f for f in run_files if tag in f and "hea" in f), None)
                    
                    if not (d_file and h_file):
                        # missing a modality, break out of this loop
                        has_all_streams = False
                        break
                
                    full_hea_path = os.path.join(root, h_file)

                    meta_df = pd.read_csv(full_hea_path)
                    fs_val = float(meta_df['Fs_Hz'].iloc[0])
                    fs_effective_val = float(meta_df["Fs_Hz_effective"].iloc[0])
                    
                    # Get sample count from header
                    count_from_header = int(meta_df['sampleCount'].iloc[0])
                    
                    # Verify actual row count in CSV file
                    actual_count = sum(1 for _ in open(os.path.join(root, d_file))) - 1
                    
                    if actual_count <= 1:
                        has_all_streams = False
                        
                    modality_files[mod] = {
                        "data": os.path.join(root, d_file),
                        "fs": fs_val,
                        "fs_effective": fs_effective_val,
                        "length": actual_count
                    }

                # Extract Subject ID for label lookup (e.g.z, 'sub-cp001') & the value
                if has_all_streams:
                    subj_str = run_files[0].split('_')[0]
                    match = subject_regex.search(subj_str)
                    subj_num = int(match.group(1))
                    
                    is_expert = expert_data[str(subj_num)]
                    
                    level_str = run_files[0]
                    levelMatch = level_regex.search(level_str)
                    level_num = int(levelMatch.group(1))
                    
                    run_num = int(run.split('_')[0])
                
                    index.append({
                        'subject': subj_str,
                        'subject_num': subj_num,
                        'run_id': run_num,
                        'level': level_num,
                        'root_folder': root,
                        'is_expert': is_expert,
                        'files': modality_files,
                    })
                    stats['valid_runs'] += 1
                else:
                    stats['skipped_runs'] += 1
                    
    index.sort(key=lambda x: (x['subject_num']))
                    
    with open(output_json, 'w') as f:
        json.dump(index, f, indent=2)
        
    return index


class CogPilotPhysiologicalDataSynchronizer:
    """
    Synchronize multimodal physiological data to 128 Hz grid
    """
    
    def __init__(self, target_fs: float = 128.0):
        self.target_fs = target_fs
        self.MATLAB_EPOCH_OFFSET = 719529
        self.SECONDS_PER_DAY = 86400
    
    def _load_and_convert_timestamps(self, filepath):
        """
        Load CSV and convert MATLAB datenum to datetime64[ns]
        """
        df = pd.read_csv(filepath)

        # Convert datenum -> unix seconds (float), rounded to microseconds
        t_adjusted = df["time_dn"].astype(np.float64).values - self.MATLAB_EPOCH_OFFSET
        t_unix_seconds = t_adjusted * self.SECONDS_PER_DAY
        t_unix_seconds = np.round(t_unix_seconds, 6)  # microsecond precision

        timestamps = pd.to_datetime(t_unix_seconds, unit="s", utc=True)
        # round index to microseconds to avoid nanosecond jitter issues
        timestamps = timestamps.round("us")

        df = df.copy()
        df["timestamps"] = timestamps
        df = df.set_index("timestamps")
        # drop original datenum column
        df = df.drop(columns=["time_dn"])
        
        # IMPORTANT: make index unique and monotonic increasing now to avoid downstream issues
        # If there are exact duplicate timestamps, keep the first (or you could aggregate with mean)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="first")]

        # Sort the index to guarantee monotonic increasing order
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        return df
    
    def _compute_actual_sampling_rate(self, df):
        """
        Calculate sampling rate from timestamps
        """
        duration = (df.index[-1] - df.index[0]).total_seconds()
        if duration <= 0:
            return 0.0
        return (len(df) - 1) / duration
    
    def _resample_array_with_polyphase(self, arr, df, source_fs):
        """
        Resample to target_fs using polyphase filtering
        """
        if arr.ndim == 1:
            arr = arr[:, None]  # make 2D

        ratio = source_fs / self.target_fs
        # If effectively the same within 1%, skip resampling at all
        if np.isclose(source_fs, self.target_fs, rtol=0.01):
            return arr.copy()

        # Find rational approx with limited denominator
        frac = Fraction(ratio).limit_denominator(1000)
        up = frac.denominator
        down = frac.numerator

        # In typical downsample case (1024 -> 128) ratio=8 => frac = 8/1 => up=1, down=8
        # If source_fs < target_fs (upsampling), up>down, works too.

        # scipy.signal.resample_poly operates along axis=0 (time axis)
        res = signal.resample_poly(arr, up=up, down=down, axis=0, window=("kaiser", 5.0))
        return res

    
    def _resample_df_to_target(self, df, source_fs):
        """
        Resample the numeric columns of df from source_fs to self.target_fs using polyphase.
        Returns a new DataFrame with a temporary index starting at df.index[0] (will be reindexed later).
        """
        if df.empty:
            return df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns to resample")

        arr = df[numeric_cols].values  # shape (n_in, n_channels)
        res_arr = self._resample_array_with_polyphase(arr, source_fs, self.target_fs)

        # Build new index aligned to an evenly spaced grid starting at df.index[0]
        n_out = res_arr.shape[0]
        start_ts = df.index[0]
        new_index = pd.date_range(start=start_ts, periods=n_out, freq=pd.Timedelta(seconds=1.0 / self.target_fs))
        res_df = pd.DataFrame(res_arr, index=new_index, columns=numeric_cols)
        return res_df
        
    def synchronize_run(self, run_files: Dict):
        """
        Synchronize all modalities for one run (intersection method)
        """
        
        loaded = {}      # mod -> (df, actual_fs)
        time_ranges = {}

        # Load & convert each modality
        for mod, info in run_files.items():
            path = info["data"]
            df = self._load_and_convert_timestamps(path)
            
            # round index so small jitter won't break comparisons
            #df.index = df.index.round("us")

            actual_fs = self._compute_actual_sampling_rate(df)

            loaded[mod] = (df, actual_fs)
            
            time_ranges[mod] = {
                                'start': df.index[0],
                                'end': df.index[-1],
                                'duration': (df.index[-1] - df.index[0]).total_seconds(),
                                'fs': actual_fs
                            }
            
        # Insert here (before computing t_start, t_end)
        """ print("=== DEBUG TIME RANGES ===")
        print("info: ", info)
        for mod, v in time_ranges.items():
            print(mod, v["start"], v["end"], v["duration"], "fs=", v["fs"])
        print("========")
         """
        # Define the master grid
        t_start = max([v["start"] for v in time_ranges.values()])
        t_end = min([v["end"] for v in time_ranges.values()])

        # build master grid once
        duration = (t_end - t_start).total_seconds()
        n_samples_target = int(np.round(duration * self.target_fs)) + 1
        
        master_grid = pd.date_range(start=t_start, periods=n_samples_target, freq=pd.Timedelta(seconds=1.0 / self.target_fs))

        aligned = {}
        
        # Align the modalities
        for mod, (df, src_fs) in loaded.items():
            # Trim to a valid window
            df_trimmed = df[(df.index >= t_start) & (df.index <= t_end)]
            #df_trimmed = df.loc[t_start:t_end]
            
            # Fallback: if no samples fall exactly inside the intersection, use nearest indices : NEW using continue
            if df_trimmed.empty:
                continue
                
            # Resample numeric columns to target
            df_resampled = self._resample_df_to_target(df_trimmed, src_fs)

            # Clean duplicates
            if df_resampled.index.duplicated().any():
                df_resampled = df_resampled[~df_resampled.index.duplicated(keep="first")]


            # WHERE WE GET THE DOUBLE ROWS FOR FINDING MEAN & STD
            # Force the timestamps to align perfectly
            df_aligned = df_resampled.reindex(master_grid, method='nearest')
            
            # Rename columns
            df_aligned.columns = [f"{mod}_{c}" for c in df_aligned.columns]
            aligned[mod] = df_aligned

        # Concatenate aligned modality dataframes along columns
        final_df = pd.concat(list(aligned.values()), axis=1)
    
        metadata = {
            "target_fs": self.target_fs,
            "time_start": master_grid[0],
            "time_end": master_grid[-1],
            "n_samples": len(master_grid),
            "modalities": list(run_files.keys()),
            "time_ranges": time_ranges,
        }

        return final_df
