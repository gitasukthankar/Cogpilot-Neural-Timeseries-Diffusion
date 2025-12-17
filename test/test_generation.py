""" """
#Quick test script to verify generation works before generating all subjects.
"""
import sys
sys.path.append(".") 

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from ntd.train_diffusion_model import init_diffusion_model
from ntd.utils.kernels_and_diffusion_utils import generate_samples
from omegaconf import OmegaConf
from hydra import initialize, compose

def test_generation():
    print("="*60)
    print("TESTING SYNTHETIC DATA GENERATION")
    print("="*60)
    
    print("\n1. Composing Hydra Config...")
    
    # We must initialize Hydra and tell it to use your overrides
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config", 
            overrides=[
                "dataset=cogpilot",
                "network=ada_conv_cogpilot",
                "diffusion=diffusion_linear_500",
                "base.experiment='debug_run'"
            ]
        )
        
    print(f"   Config loaded.")
    print(f"   Signal Channels: {cfg.network.signal_channel}") 
    print(f"   Signal Length:   {cfg.dataset.signal_length}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        map_loc = None # Default behavior
    else:
        device = torch.device("cpu")
        map_loc = torch.device("cpu")
    print(f"   Device: {device}")
    print(f"   Map Location: {map_loc}")
    
    # Load model
    print("\n2. Loading trained model...")
    diffusion, network = init_diffusion_model(cfg)
    diffusion = diffusion.to(device)
    
    # Load model
    experiment_name = cfg.base.experiment  # debug_run
    model_filename = f"{experiment_name}_models.pkl"
    model_path = Path(experiment_name) / model_filename
    
    state_dict = pickle.load(open(model_path, "rb"))
    diffusion.load_state_dict(state_dict)
    print(f"   ✓ Model loaded from {model_path} and on {device}")
    
    # Load normalization stats
    print("\n3. Loading normalization statistics...")
    stats_path = "cogpilot_global_stats.pkl"
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    ch_mean = stats['mean']
    ch_std = stats['std']
    
    # Convert to numpy
    if torch.is_tensor(ch_mean):
        ch_mean_np = ch_mean.cpu().numpy().squeeze()
        ch_std_np = ch_std.cpu().numpy().squeeze()
    else:
        ch_mean_np = ch_mean.squeeze()
        ch_std_np = ch_std.squeeze()
    
    print(f"   Mean shape: {ch_mean_np.shape}  Std shape: {ch_std_np.shape}")
    print(f"   Mean ECG_LL_RA (ECG): {ch_mean_np[0]:.2f}")
    print(f"   Std ECG_LL_RA (ECG): {ch_std_np[0]:.2f}")
    print("\n\nExample channel means/std:", mean_np[:3], std_np[:3])
    
    # Generate samples
    print("\n4. Generating sample windows...")
    
    # Generate expert sample
    # Conditioning: 1.0 for Expert
    # Shape: (Batch, Cond_Dim, Time) -> (4, 1, 512)
    cond_expert = torch.ones(4, 1, 512).to(device) * 1.0  # Expert
    with torch.no_grad():
        samples_expert = generate_samples(
            diffusion=diffusion,
            total_num_samples=4,
            batch_size=4,
            cond=cond_expert
        )
    
    # Generate novice sample
    cond_novice = torch.ones(4, 1, 512).to(device) * 0.0  # Novice
    with torch.no_grad():
        samples_novice = generate_samples(
            diffusion=diffusion,
            total_num_samples=4,
            batch_size=4,
            cond=cond_novice
        )
    
    print(f"   Expert samples shape: {samples_expert.shape}")
    print(f"   Novice samples shape: {samples_novice.shape}")
    print(f"   Expert mean: {samples_expert.mean().item():.3f}, std: {samples_expert.std().item():.3f}")
    print(f"   Novice mean: {samples_novice.mean().item():.3f}, std: {samples_novice.std().item():.3f}")
    
    
    # Check if normalized
    if abs(samples_expert.mean().item()) < 0.5 and abs(samples_expert.std().item() - 1.0) < 0.5:
        print(f"   ✓ Samples are NORMALIZED (need denormalization)")
    else:
        print(f"   ⚠ Samples don't look normalized")
    
    # Denormalize
    print("\n5. Denormalizing samples...")
    
    # Take first window from each
    expert_window = samples_expert[0].cpu().numpy()  # (14, 512)
    novice_window = samples_novice[0].cpu().numpy()  # (14, 512)
    
    print(f"   Expert window shape: {expert_window.shape}")
    print(f"   Stats shape: ch_mean={ch_mean_np.shape}, ch_std={ch_std_np.shape}")
    print(f"   Expert window ECG_LL_RA (ECG) stats: mean={expert_window[0].mean():.3f}, std={expert_window[0].std():.3f}")
    
    """ # Denormalize: (14, 512) * (14, 1) + (14, 1)
   # expert_denorm = expert_window * ch_std_np[:, None] + ch_mean_np[:, None]
    #novice_denorm = novice_window * ch_std_np[:, None] + ch_mean_np[:, None] 
"""
    
    # Check if already denormalized (if mean/std match training stats)
    expert_ch0_mean = expert_window[0].mean()
    expected_ch0_mean = ch_mean_np[0]
    
    if abs(expert_ch0_mean - expected_ch0_mean) < 5.0:
        print(f"   ⚠️  WARNING: Data might already be denormalized!")
        print(f"      Expected normalized mean ≈ 0, got {expert_ch0_mean:.3f}")
        print(f"      Training mean = {expected_ch0_mean:.3f}")
        expert_denorm = expert_window
        novice_denorm = novice_window
    else:
        # Denormalize: (14, 512) * (14, 1) + (14, 1)
        expert_denorm = expert_window * ch_std_np[:, None] + ch_mean_np[:, None]
        novice_denorm = novice_window * ch_std_np[:, None] + ch_mean_np[:, None]
    
    """# Reshape stats for broadcasting
    # Stats are (14,), we need (14, 1) to multiply against (14, 512)
    #mean_vec = ch_mean_np[:, None]
    #std_vec = ch_std_np[:, None]

    # FORCE Denormalization (No if-checks!)
    # Formula: Real = (Normalized * Std) + Mean
    #expert_denorm = expert_window * std_vec + mean_vec
    #novice_denorm = novice_window * std_vec + mean_vec 
"""
    
    print(f"   Expert ECG_LL_RA after denorm: mean={expert_denorm[0].mean():.2f}, std={expert_denorm[0].std():.2f}")
    print(f"   Novice ECG_LL_RA after denorm: mean={novice_denorm[0].mean():.2f}, std={novice_denorm[0].std():.2f}")
    print(f"   Expert ECG_LL_RA range: [{expert_denorm[0].min():.2f}, {expert_denorm[0].max():.2f}]")
    print(f"   Novice ECG_LL_RA range: [{novice_denorm[0].min():.2f}, {novice_denorm[0].max():.2f}]")
    
    # Compare to expected training stats
    print(f"\n   Expected from training:")
    print(f"   ECG_LL_RA mean: {ch_mean_np[0]:.2f}, std: {ch_std_np[0]:.2f}")
    print(f"   ECG_LA_RA mean: {ch_mean_np[1]:.2f}, std: {ch_std_np[1]:.2f}")
    print(f"   ECG_VX_RL mean: {ch_mean_np[2]:.2f}, std: {ch_std_np[2]:.2f}")
    
    # Visualize
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    channel_names = ['ECG_LL_RA', 'ECG_LA_RA', 'ECG_VX_RL']
    
    for i, name in enumerate(channel_names):
        # Normalized
        axes[i, 0].plot(expert_window[i], label='Expert', alpha=0.7)
        axes[i, 0].plot(novice_window[i], label='Novice', alpha=0.7)
        axes[i, 0].set_title(f'{name} - Normalized')
        axes[i, 0].set_ylabel('Normalized Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Denormalized
        axes[i, 1].plot(expert_denorm[i], label='Expert', alpha=0.7)
        axes[i, 1].plot(novice_denorm[i], label='Novice', alpha=0.7)
        axes[i, 1].set_title(f'{name} - Denormalized')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time (samples)')
    axes[2, 1].set_xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.savefig('test_generation_output.png', dpi=150)
    print(f"   ✓ Saved visualization to test_generation_output.png")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check test_generation_output.png")
    print("2. Verify ECG signals show heartbeat patterns")
    print("3. If looks good, run full generation:")
    print("   python3 generate_cogpilot.py ...")
    
    plt.show()

if __name__ == "__main__":
    test_generation() """
    
"""
Diagnostic test script to identify denormalization issues.
"""
import sys
sys.path.append(".") 

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from ntd.train_diffusion_model import init_diffusion_model
from ntd.utils.kernels_and_diffusion_utils import generate_samples
from omegaconf import OmegaConf
from hydra import initialize, compose

def test_generation():
    print("="*70)
    print("DIAGNOSTIC: TESTING DENORMALIZATION")
    print("="*70)
    
    print("\n[1/7] Loading Hydra Config...")
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config", 
            overrides=[
                "dataset=cogpilot",
                "network=ada_conv_cogpilot",
                "diffusion=diffusion_linear_500",
                "base.experiment='debug_run'"
            ]
        )
    
    print(f"   ✓ Signal Channels: {cfg.network.signal_channel}") 
    print(f"   ✓ Signal Length: {cfg.dataset.signal_length}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Device: {device}")
    
    # Load model
    print("\n[2/7] Loading Trained Model...")
    diffusion, network = init_diffusion_model(cfg)
    diffusion = diffusion.to(device)
    
    experiment_name = cfg.base.experiment
    model_filename = f"{experiment_name}_models.pkl"
    model_path = Path(experiment_name) / model_filename
    
    if not model_path.exists():
        print(f"   ❌ ERROR: Model not found at {model_path}")
        return
    
    state_dict = pickle.load(open(model_path, "rb"))
    diffusion.load_state_dict(state_dict)
    print(f"   ✓ Model loaded from {model_path}")
    
    # Load normalization stats
    print("\n[3/7] Loading Normalization Statistics...")
    stats_path = "cogpilot_global_stats.pkl"
    
    if not Path(stats_path).exists():
        print(f"   ❌ ERROR: Stats file not found at {stats_path}")
        return
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Extract and convert stats
    ch_mean = stats['mean']
    ch_std = stats['std']
    
    print(f"   Stats type: {type(ch_mean)}")
    print(f"   Stats device: {ch_mean.device if torch.is_tensor(ch_mean) else 'numpy'}")
    
    # Convert to numpy for analysis
    if torch.is_tensor(ch_mean):
        ch_mean_np = ch_mean.cpu().numpy()
        ch_std_np = ch_std.cpu().numpy()
    else:
        ch_mean_np = ch_mean
        ch_std_np = ch_std
    
    # Flatten if needed
    if ch_mean_np.ndim > 1:
        ch_mean_np = ch_mean_np.flatten()
        ch_std_np = ch_std_np.flatten()
    
    print(f"   ✓ Stats shape: mean={ch_mean_np.shape}, std={ch_std_np.shape}")
    
    # Print expected stats
    channel_names = ['ECG_LL_RA', 'ECG_LA_RA', 'ECG_VX_RL', 'PPG', 'EDA',
                     'ACC_FA_X', 'ACC_FA_Y', 'ACC_FA_Z', 'EMG_FLEX', 'EMG_EXT',
                     'RESP', 'ACC_T_X', 'ACC_T_Y', 'ACC_T_Z']
    
    print("\n   Expected Training Statistics:")
    print("   " + "-"*60)
    for i in range(min(3, len(channel_names))):  # Just ECG channels
        print(f"   {channel_names[i]:12s}: mean={ch_mean_np[i]:8.2f}, std={ch_std_np[i]:8.2f}")
    print("   " + "-"*60)
    
    # Generate samples
    print("\n[4/7] Generating Test Samples...")
    
    # Test with unified label conditioning
    # Label 0 = Novice Level 1, Label 4 = Expert Level 1
    signal_length = cfg.dataset.signal_length
    
    # Generate Novice Level 1 (label=0)
    cond_novice = torch.full((2, 1, signal_length), 0.0, device=device)
    with torch.no_grad():
        samples_novice = generate_samples(
            diffusion=diffusion,
            total_num_samples=2,
            batch_size=2,
            cond=cond_novice
        )
    
    # Generate Expert Level 1 (label=4)
    cond_expert = torch.full((2, 1, signal_length), 4.0, device=device)
    with torch.no_grad():
        samples_expert = generate_samples(
            diffusion=diffusion,
            total_num_samples=2,
            batch_size=2,
            cond=cond_expert
        )
    
    print(f"   ✓ Generated shapes: {samples_novice.shape}, {samples_expert.shape}")
    
    # Move to CPU and get first sample
    novice_window = samples_novice[0].cpu().numpy()  # (14, 512)
    expert_window = samples_expert[0].cpu().numpy()  # (14, 512)
    
    # Analyze raw generated samples
    print("\n[5/7] Analyzing Raw Generated Samples (Should be Normalized)...")
    print("   " + "-"*60)
    print(f"   Novice sample:")
    print(f"     Overall: mean={novice_window.mean():.4f}, std={novice_window.std():.4f}")
    print(f"     ECG_LL_RA: mean={novice_window[0].mean():.4f}, std={novice_window[0].std():.4f}")
    print(f"     ECG_LA_RA: mean={novice_window[1].mean():.4f}, std={novice_window[1].std():.4f}")
    print(f"     ECG_VX_RL: mean={novice_window[2].mean():.4f}, std={novice_window[2].std():.4f}")
    
    print(f"\n   Expert sample:")
    print(f"     Overall: mean={expert_window.mean():.4f}, std={expert_window.std():.4f}")
    print(f"     ECG_LL_RA: mean={expert_window[0].mean():.4f}, std={expert_window[0].std():.4f}")
    
    # Check if normalized
    is_normalized = abs(novice_window.mean()) < 1.0 and abs(novice_window.std() - 1.0) < 1.0
    
    if is_normalized:
        print(f"\n   ✓ Samples appear NORMALIZED (mean≈0, std≈1)")
        print(f"     This is CORRECT - they need denormalization")
    else:
        print(f"\n   ⚠️  Samples DON'T look normalized!")
        print(f"     Expected: mean≈0, std≈1")
        print(f"     Got: mean={novice_window.mean():.3f}, std={novice_window.std():.3f}")
    
    print("   " + "-"*60)
    
    # Denormalize
    print("\n[6/7] Applying Denormalization...")
    print("   Formula: x_real = (x_normalized * std) + mean")
    
    # Reshape stats for broadcasting: (14,) -> (14, 1)
    mean_vec = ch_mean_np[:, None]  # (14, 1)
    std_vec = ch_std_np[:, None]    # (14, 1)
    
    print(f"   Stats for broadcasting: mean={mean_vec.shape}, std={std_vec.shape}")
    print(f"   Sample shape: {novice_window.shape}")
    
    # Denormalize
    novice_denorm = (novice_window * std_vec) + mean_vec
    expert_denorm = (expert_window * std_vec) + mean_vec
    
    print("\n   After Denormalization:")
    print("   " + "-"*60)
    print(f"   Novice ECG_LL_RA:")
    print(f"     mean = {novice_denorm[0].mean():8.2f} (expected: {ch_mean_np[0]:8.2f})")
    print(f"     std  = {novice_denorm[0].std():8.2f} (expected: {ch_std_np[0]:8.2f})")
    print(f"     range = [{novice_denorm[0].min():8.2f}, {novice_denorm[0].max():8.2f}]")
    
    print(f"\n   Expert ECG_LL_RA:")
    print(f"     mean = {expert_denorm[0].mean():8.2f}")
    print(f"     std  = {expert_denorm[0].std():8.2f}")
    print(f"     range = [{expert_denorm[0].min():8.2f}, {expert_denorm[0].max():8.2f}]")
    
    print(f"\n   Novice ECG_VX_RL:")
    print(f"     mean = {novice_denorm[2].mean():8.2f} (expected: {ch_mean_np[2]:8.2f})")
    print(f"     std  = {novice_denorm[2].std():8.2f} (expected: {ch_std_np[2]:8.2f})")
    print(f"     range = [{novice_denorm[2].min():8.2f}, {novice_denorm[2].max():8.2f}]")
    print("   " + "-"*60)
    
    # Diagnostic checks
    print("\n[7/7] DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues_found = []
    
    # Check 1: Are means close to expected?
    mean_error_ch0 = abs(novice_denorm[0].mean() - ch_mean_np[0])
    if mean_error_ch0 > 50:  # Tolerance of 50 mV
        issues_found.append(f"❌ ECG_LL_RA mean off by {mean_error_ch0:.1f} mV")
    else:
        print(f"✓ ECG_LL_RA mean is correct (error: {mean_error_ch0:.1f} mV)")
    
    # Check 2: Are stds reasonable?
    std_ratio_ch0 = novice_denorm[0].std() / ch_std_np[0]
    if std_ratio_ch0 < 0.5 or std_ratio_ch0 > 2.0:
        issues_found.append(f"❌ ECG_LL_RA std ratio is {std_ratio_ch0:.2f} (should be ~1.0)")
    else:
        print(f"✓ ECG_LL_RA std is reasonable (ratio: {std_ratio_ch0:.2f})")
    
    # Check 3: Is there drift?
    first_half_mean = novice_denorm[2, :256].mean()
    second_half_mean = novice_denorm[2, 256:].mean()
    drift = abs(second_half_mean - first_half_mean)
    if drift > 10:  # More than 10 mV drift in 2 seconds is suspicious
        issues_found.append(f"❌ ECG_VX_RL shows drift: {drift:.1f} mV over 2 seconds")
    else:
        print(f"✓ No significant drift detected ({drift:.1f} mV)")
    
    # Check 4: Are values in realistic physiological range?
    ecg_range = novice_denorm[0].max() - novice_denorm[0].min()
    if ecg_range < 0.1 or ecg_range > 100:  # ECG range should be ~0.5-10 mV
        issues_found.append(f"❌ ECG range unusual: {ecg_range:.2f} mV")
    else:
        print(f"✓ ECG range is realistic ({ecg_range:.2f} mV)")
    
    print("="*70)
    
    if issues_found:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues_found:
            print(f"   {issue}")
        print("\nPossible causes:")
        print("  1. Model trained on subset of data (not seeing full distribution)")
        print("  2. Stats file doesn't match actual training data")
        print("  3. Model not trained long enough (150 epochs on full data)")
        print("  4. Bug in denormalization code")
    else:
        print("\n✓ ALL CHECKS PASSED!")
        print("  Denormalization appears to be working correctly.")
    
    # Visualize
    print("\n[BONUS] Creating Diagnostic Plots...")
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    time_axis = np.arange(512) / 128.0  # Convert to seconds
    
    # Row 1: ECG_LL_RA
    axes[0, 0].plot(time_axis, novice_window[0], label='Novice', alpha=0.8)
    axes[0, 0].plot(time_axis, expert_window[0], label='Expert', alpha=0.8)
    axes[0, 0].set_title('ECG_LL_RA - NORMALIZED (Raw Output)')
    axes[0, 0].set_ylabel('Normalized Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[0, 1].plot(time_axis, novice_denorm[0], label='Novice', alpha=0.8)
    axes[0, 1].plot(time_axis, expert_denorm[0], label='Expert', alpha=0.8)
    axes[0, 1].set_title('ECG_LL_RA - DENORMALIZED')
    axes[0, 1].set_ylabel('mV')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(ch_mean_np[0], color='r', linestyle='--', alpha=0.5, label='Expected Mean')
    
    # Row 2: ECG_LA_RA
    axes[1, 0].plot(time_axis, novice_window[1], alpha=0.8)
    axes[1, 0].set_title('ECG_LA_RA - NORMALIZED')
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_axis, novice_denorm[1], alpha=0.8)
    axes[1, 1].set_title('ECG_LA_RA - DENORMALIZED')
    axes[1, 1].set_ylabel('mV')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(ch_mean_np[1], color='r', linestyle='--', alpha=0.5)
    
    # Row 3: ECG_VX_RL (Check for drift)
    axes[2, 0].plot(time_axis, novice_window[2], alpha=0.8)
    axes[2, 0].set_title('ECG_VX_RL - NORMALIZED')
    axes[2, 0].set_ylabel('Normalized Value')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time_axis, novice_denorm[2], alpha=0.8)
    axes[2, 1].set_title('ECG_VX_RL - DENORMALIZED (Check for Drift)')
    axes[2, 1].set_ylabel('mV')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(ch_mean_np[2], color='r', linestyle='--', alpha=0.5)
    # Mark first and second half means
    axes[2, 1].axhline(first_half_mean, color='g', linestyle=':', alpha=0.5, label='1st half')
    axes[2, 1].axhline(second_half_mean, color='orange', linestyle=':', alpha=0.5, label='2nd half')
    axes[2, 1].legend()
    
    # Row 4: Distribution comparison
    axes[3, 0].hist(novice_window[0], bins=50, alpha=0.7, label='Novice')
    axes[3, 0].hist(expert_window[0], bins=50, alpha=0.7, label='Expert')
    axes[3, 0].set_title('ECG_LL_RA Distribution - NORMALIZED')
    axes[3, 0].set_xlabel('Value')
    axes[3, 0].set_ylabel('Count')
    axes[3, 0].legend()
    axes[3, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    
    axes[3, 1].hist(novice_denorm[0], bins=50, alpha=0.7, label='Novice')
    axes[3, 1].hist(expert_denorm[0], bins=50, alpha=0.7, label='Expert')
    axes[3, 1].set_title('ECG_LL_RA Distribution - DENORMALIZED')
    axes[3, 1].set_xlabel('mV')
    axes[3, 1].set_ylabel('Count')
    axes[3, 1].legend()
    axes[3, 1].axvline(ch_mean_np[0], color='r', linestyle='--', alpha=0.5)
    
    for ax in axes.flat:
        ax.label_outer()
    
    plt.tight_layout()
    plt.savefig('diagnostic_denormalization.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved diagnostic plot to: diagnostic_denormalization.png")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check diagnostic_denormalization.png for visual inspection")
    print("2. If issues found, retrain on FULL dataset (not debug subset)")
    print("3. Verify stats file matches training data")
    print("4. Run full generation only if checks pass")
    print("="*70)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    test_generation()