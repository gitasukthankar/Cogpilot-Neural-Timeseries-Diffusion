import hydra
import torch
import numpy as np
import pandas as pd
import pickle
import os
from omegaconf import OmegaConf
from ntd.train_diffusion_model import init_diffusion_model
from ntd.utils.kernels_and_diffusion_utils import generate_samples
from pathlib import Path
from datetime import datetime, timedelta
import logging
log = logging.getLogger(__name__)

CHANNEL_MAP = {
    'ecg':  slice(0, 3),
    'eda':  slice(3, 5),
    'emg':  slice(5, 10),
    'resp': slice(10, 11),
    'acc':  slice(11, 14)
}

FILE_SUFFIXES = {
    'ecg':  'stream-lslshimmerecg',
    'eda':  'stream-lslshimmereda',
    'emg':  'stream-lslshimmeremg',
    'resp': 'stream-lslshimmerresp',
    'acc':  'stream-lslshimmertorsoacc'
}

COL_NAMES = {
    'ecg':  ['ecg_projection_ll_ra_mV', 'ecg_projection_la_ra_mV', 'ecg_projection_vx_rl_mV'],
    'eda':  ['ppg_finger_mV', 'eda_hand_l_kOhms'],
    'emg':  ['accelerometry_forearm_r_x_mps2', 'accelerometry_forearm_r_y_mps2', 'accelerometry_forearm_r_z_mps2', 'emg_wrist_flexor_mV', 'emg_wrist_extensor_mV'],
    'resp': ['respiration_trace_mV'],
    'acc':  ['accelerometry_torso_x_mps2', 'accelerometry_torso_y_mps2', 'accelerometry_torso_z_mps2']
}

LEVEL_SEQUENCE = [
    "01B", "03B", "02B", "04B",  # Runs 1-4
    "03B", "04B", "01B", "02B",  # Runs 5-8
    "04B", "02B", "03B", "01B"   # Runs 9-12
]

# Level mapping: matches your dataset unified_label calculation
# Formula: label = (is_expert * 4) + (difficulty - 1)
LEVEL_MAP = {
    0: {"expertise": "Novice", "level": 1, "level_code": "01B"},
    1: {"expertise": "Novice", "level": 2, "level_code": "02B"},
    2: {"expertise": "Novice", "level": 3, "level_code": "03B"},
    3: {"expertise": "Novice", "level": 4, "level_code": "04B"},
    4: {"expertise": "Expert", "level": 1, "level_code": "01B"},
    5: {"expertise": "Expert", "level": 2, "level_code": "02B"},
    6: {"expertise": "Expert", "level": 3, "level_code": "03B"},
    7: {"expertise": "Expert", "level": 4, "level_code": "04B"},
}

def decode_label(label_index):
    """Helper to decode unified label to human-readable format."""
    return LEVEL_MAP.get(int(label_index), {"expertise": "Unknown", "level": 0, "level_code": "00B"})


OUTPUT_ROOT = Path("generated_data_root")
NUM_SUBJECTS = 20
WINDOWS_PER_RUN = 1
START_DT = datetime(2021, 1, 1, 12, 0, 0)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate(cfg):
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

    # Load trained model
    diffusion, network = init_diffusion_model(cfg)
    diffusion = diffusion.to(device)
    
    experiment_name = cfg.base.experiment  # debug_run
    model_filename = f"{experiment_name}_models.pkl"
    model_path = Path(experiment_name) / model_filename
    
    if os.path.exists(model_path):
        log.info(f"Loading model from {model_path}")
        state_dict = pickle.load(open(model_path, "rb"))
        diffusion.load_state_dict(state_dict)
    else:
        log.info("Warning: Model weights not found. Generating with random weights.")
    
    # Load global stats for unnormalization
    stats_path = 'cogpilot_global_stats.pkl'
    if os.path.exists(stats_path):
        log.info("Stats file found")
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
            # Load mean/std and move to CPU numpy for calculation
            # Shape is likely (14, 1) or (14,)
            global_mean = stats['mean'].cpu().numpy().flatten()
            global_std = stats['std'].cpu().numpy().flatten()
            
            # Reshape for broadcasting: (14, 1) to match (Channels, Time)
            global_mean = global_mean[:, None]
            global_std = global_std[:, None]
    else:
        log.info("Stats file not found. Generated data will be standardized")
        global_mean = 0.0
        global_std = 1.0
        
        
    signal_length = cfg.dataset.signal_length
    fs = cfg.dataset.target_fs
    samples_per_run = WINDOWS_PER_RUN * signal_length
    
    for subject_index in range(1, NUM_SUBJECTS + 1):
        current_dt = START_DT
        
        # Determine expertise: even IDs = Expert, odd IDs = Novice
        is_expert = 1 if subject_index % 2 == 0 else 0
        expertise_type = "Expert" if is_expert else "Novice"
        
        subject_id = f'sub-cp{subject_index:03d}'
        
        log.info("")
        log.info(f"Generating {subject_id} ({expertise_type})")
        log.info("-" * 70)
        
        session_id = "ses-20230101"
        
        # Generate 12 runs (4 levels × 3 blocks)
        run_counter = 1
        
        for block_idx in range(3):  # 3 blocks
            log.info(f"  Block {block_idx + 1}/3:")
            
            for level_idx in range(4):  # 4 levels per block
                # Get level for this run from the sequence
                global_run_idx = block_idx * 4 + level_idx
                level_code = LEVEL_SEQUENCE[global_run_idx]
                
                # Convert level_code to difficulty number
                difficulty = int(level_code[1])  # "01B" -> 1, "04B" -> 4
                
                # Calculate unified label (matching dataset logic)
                # Formula: label = (is_expert * 4) + (difficulty - 1)
                unified_label = (is_expert * 4) + (difficulty - 1)
                
                # Verify it matches our LEVEL_MAP
                level_info = LEVEL_MAP[unified_label]
                assert level_info['level'] == difficulty, "Label mismatch!"
                assert level_info['expertise'] == expertise_type, "Expertise mismatch!"
                
                run_name = f"level-{level_code}_run-{run_counter:03d}"
                run_path = OUTPUT_ROOT / "task-ils" / subject_id / session_id / run_name
                run_path.mkdir(parents=True, exist_ok=True)
                
                # Create conditioning tensor for this specific run
                # Shape: (batch=1, cond_dim=1, length=512)
                # AdaConv's GeneralEmbedder expects 3D: (batch, cond_dim, length)
                cond_tensor = torch.full(
                    (WINDOWS_PER_RUN, 1, signal_length), 
                    float(unified_label), 
                    dtype=torch.float32,
                    device=device
                )
                
                log.info(f"    Run {run_counter:02d}: Level {level_code} (label={unified_label}, {expertise_type})")
                
                # GENERATE SYNTHETIC DATA
                with torch.no_grad():
                    samples = generate_samples(
                        diffusion=diffusion,
                        total_num_samples=WINDOWS_PER_RUN,
                        batch_size=WINDOWS_PER_RUN,
                        cond=cond_tensor
                    )
                    # samples shape: (batch=1, channels=14, time=512)
                
                # Un-normalize data
                # samples is on GPU, move to CPU
                samples_cpu = samples.cpu().numpy()  # (1, 14, 512)
                
                # Remove batch dimension since WINDOWS_PER_RUN=1
                signal_norm = samples_cpu[0]  # (14, 512)
                
                # Apply inverse normalization
                signal_raw = (signal_norm * global_std) + global_mean  # (14, 512) -> has to change as we now have a shape of (14, 1)
                
                # Transpose to (Time, Channels) for CSV
                signal_final = signal_raw.T  # (512, 14)
                
                # Create time column
                time_deltas = pd.to_timedelta(np.arange(samples_per_run) / fs, unit='s')
                time_col = current_dt + time_deltas
                current_dt = time_col[-1] + pd.Timedelta(seconds=5)  # 5s gap between runs
                
                # SAVE INDIVIDUAL CSVs FOR EACH MODALITY
                for modality, channel_slice in CHANNEL_MAP.items():
                    # Extract columns for this modality
                    mod_data = signal_final[:, channel_slice]
                    
                    # Construct filename
                    base_fname = f"{subject_id}_{session_id}_task-ils_{FILE_SUFFIXES[modality]}_feat-chunk_{run_name}"
                    dat_file = run_path / f"{base_fname}_dat.csv"
                    hea_file = run_path / f"{base_fname}_hea.csv"
                    
                    # Save data CSV with time column
                    df = pd.DataFrame(mod_data, columns=COL_NAMES[modality])
                    df.insert(0, 'time_dn', time_col)
                    df.to_csv(dat_file, index=False)
                    
                    # Save header CSV with metadata
                    hea_df = pd.DataFrame({
                        'Fs_Hz': [fs],
                        'duration_s': [samples_per_run / fs],
                        'num_samples': [samples_per_run],
                        'expertise': [expertise_type],
                        'level': [difficulty],
                        'unified_label': [unified_label]
                    })
                    hea_df.to_csv(hea_file, index=False)
                
                run_counter += 1
    
    """ # Generate Subjects
    for subject_index in range(1, NUM_SUBJECTS+1):
        current_dt = START_DT
        
        # conditioning, even IDs are experts, odds are novince
        is_expert = 1.0 if subject_index % 2 == 0 else 0.0
        subject_type = "Expert" if is_expert else "Novince"
        
        subject_id = f'sub-cp{subject_index:03d}'
        
        log.info(f"Generating {subject_id}")
        
        # Create Conditioning Tensor for this subject
        # shape: (batch, 1) for NTD AdaConv
        
        ###new cond_tensor, as the AdaConv expects this shape: (batch_size, cond_dim)
        ###cond_tensor = torch.ones(WINDOWS_PER_RUN, 1, signal_length).to(device) * is_expert
        #cond_tensor = torch.full((WINDOWS_PER_RUN, 1), is_expert, device=device)

        #Session ID
        session_id = "ses-20230101"
        
        # Generate 4 Levels x 4 Runs 
        run_counter = 1
        levels = ["01B", "03B", "02B", "04B"]
        
        for block_idx in range(3):  # 3 blocks
            log.info(f"  Block {block_idx + 1}/3:")
            
            for level_idx in range(4):  # 4 levels per block
                # Get level for this run from the sequence
                global_run_idx = block_idx * 4 + level_idx
                level_code = LEVEL_SEQUENCE[global_run_idx]
                
                # Convert level_code to difficulty number
                difficulty = int(level_code[1])  # "01B" -> 1, "04B" -> 4
                
                # Calculate unified label (matching dataset logic)
                # Formula: label = (is_expert * 4) + (difficulty - 1)
                unified_label = (is_expert * 4) + (difficulty - 1)
                
                # Verify it matches our LEVEL_MAP
                level_info = LEVEL_MAP[unified_label]
                assert level_info['level'] == difficulty, "Label mismatch!"
                assert level_info['expertise'] == expertise_type, "Expertise mismatch!"
                
                run_name = f"level-{level_code}_run-{run_counter:03d}"
                run_path = OUTPUT_ROOT / "task-ils" / subject_id / session_id / run_name
                run_path.mkdir(parents=True, exist_ok=True)
                
                # Create conditioning tensor for this specific run
                # Shape: (batch=1, cond_dim=1, length=512)
                # AdaConv's GeneralEmbedder expects 3D: (batch, cond_dim, length)
                cond_tensor = torch.full(
                    (WINDOWS_PER_RUN, 1, signal_length), 
                    float(unified_label), 
                    dtype=torch.float32,
                    device=device
                )
                
                log.info(f"    Run {run_counter:02d}: Level {level_code} (label={unified_label}, {expertise_type})")
                
                # GENERATE SYNTHETIC DATA
                with torch.no_grad():
                    samples = generate_samples(
                        diffusion=diffusion,
                        total_num_samples=WINDOWS_PER_RUN,
                        batch_size=WINDOWS_PER_RUN,
                        cond=cond_tensor
                    )
                    # samples shape: (batch=1, channels=14, time=512)
                
                # Un-normalize data
                # samples is on GPU, move to CPU
                samples_cpu = samples.cpu().numpy()  # (1, 14, 512)
                
                # Remove batch dimension since WINDOWS_PER_RUN=1
                signal_norm = samples_cpu[0]  # (14, 512)
                
                # Apply inverse normalization
                signal_raw = (signal_norm * global_std) + global_mean  # (14, 512)
                
                # Transpose to (Time, Channels) for CSV
                signal_final = signal_raw.T  # (512, 14)
                
                # Create time column
                time_deltas = pd.to_timedelta(np.arange(samples_per_run) / fs, unit='s')
                time_col = current_dt + time_deltas
                current_dt = time_col[-1] + pd.Timedelta(seconds=5)  # 5s gap between runs
                
                # SAVE INDIVIDUAL CSVs FOR EACH MODALITY
                for modality, channel_slice in CHANNEL_MAP.items():
                    # Extract columns for this modality
                    mod_data = signal_final[:, channel_slice]
                    
                    # Construct filename
                    base_fname = f"{subject_id}_{session_id}_task-ils_{FILE_SUFFIXES[modality]}_feat-chunk_{run_name}"
                    dat_file = run_path / f"{base_fname}_dat.csv"
                    hea_file = run_path / f"{base_fname}_hea.csv"
                    
                    # Save data CSV with time column
                    df = pd.DataFrame(mod_data, columns=COL_NAMES[modality])
                    df.insert(0, 'time_dn', time_col)
                    df.to_csv(dat_file, index=False)
                    
                    # Save header CSV with metadata
                    hea_df = pd.DataFrame({
                        'Fs_Hz': [fs],
                        'duration_s': [samples_per_run / fs],
                        'num_samples': [samples_per_run],
                        'expertise': [expertise_type],
                        'level': [difficulty],
                        'unified_label': [unified_label]
                    })
                    hea_df.to_csv(hea_file, index=False)
                
                run_counter += 1 """
            
    print("\nGeneration Complete!")
    print(f"Data saved to: {OUTPUT_ROOT.resolve()}")
  
if __name__ == "__main__":
    generate()
