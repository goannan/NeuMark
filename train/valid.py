import torch
import torchaudio
import random
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add the project root and train directories to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from infer.watermark import get_solver
from attacks import AudioEffects
from attacks.augmentation import apply_encodec, apply_dac, apply_wavtokenizer
from quality import compute_quality

# 5. Run multi-dimensional validation
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-ck', type=str, required=True, help='Checkpoint path')
parser.add_argument('--dataset_root', '-d', type=str, default=os.path.join(project_root, "dataset/LibriSpeech/test-clean"), help='Dataset root folder')
parser.add_argument('--output_dir', '-o', type=str, default=os.path.join(project_root, "valid_samples"), help='Output folder for audio samples')
parser.add_argument('--config', '-c', type=str, default=os.path.join(project_root, "config/default.json"), help='Path to configuration JSON file')
args = parser.parse_args()

# 1. Environment configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Prepare dataset file list
dataset_root = Path(args.dataset_root)
print(f"Scanning dataset: {dataset_root} ...")
if not dataset_root.exists():
    print(f"Error: Dataset directory {dataset_root} not found")
    sys.exit(1)

# Get file list (recursively search for flac or wav)
all_wav_files = sorted(list(dataset_root.rglob("*.flac")) + list(dataset_root.rglob("*.wav")))

test_files = all_wav_files
print(f"Scanned {len(all_wav_files)} files from {dataset_root.name}, using {len(test_files)} samples for validation.")

# 4. Define all validation attacks
test_attacks = {
    "Clean (Identity)": lambda x, sr: AudioEffects.identity(x),
    "Gaussian Noise": lambda x, sr: AudioEffects.random_noise(x, noise_std=0.001),
    "Pink Noise": lambda x, sr: AudioEffects.pink_noise(x, noise_std=0.01),
    "Lowpass Filter (5k)": lambda x, sr: AudioEffects.lowpass_filter(x, cutoff_freq=5000, sample_rate=sr),
    "Highpass Filter (500)": lambda x, sr: AudioEffects.highpass_filter(x, cutoff_freq=500, sample_rate=sr),
    "Bandpass Filter": lambda x, sr: AudioEffects.bandpass_filter(x, cutoff_freq_low=300, cutoff_freq_high=8000, sample_rate=sr),
    "Echo/Reverb": lambda x, sr: AudioEffects.echo(x, volume_range=(0.1, 0.5), duration_range=(0.1, 0.5), sample_rate=sr),
    "Smooth (Moving Avg)": lambda x, sr: AudioEffects.smooth(x, window_size_range=(2, 10)),
    "Resampling (32k-16k)": lambda x, sr: AudioEffects.updownresample(x, sample_rate=sr, intermediate_freq=32000),
    "Volume Boost (+10%)": lambda x, sr: AudioEffects.boost_audio(x, amount=10),
    "Volume Duck (-10%)": lambda x, sr: AudioEffects.duck_audio(x, amount=10),
    "Speed (0.8x-1.2x)": lambda x, sr: AudioEffects.speed(x, speed_range=(0.8, 1.2), sample_rate=sr),
    "Encodec 3.0kbps": lambda x, sr: apply_encodec(x, sr, 3.0),
    "Encodec 6.0kbps": lambda x, sr: apply_encodec(x, sr, 6.0),
    "Encodec 12.0kbps": lambda x, sr: apply_encodec(x, sr, 12.0),
    "DAC 16kHz": lambda x, sr: apply_dac(x, sr, 16000),
    "DAC 24kHz": lambda x, sr: apply_dac(x, sr, 24000),
    "WavTokenizer": lambda x, sr: apply_wavtokenizer(x, sr),
}

# 5. Run multi-dimensional validation
summary_report = {}
quality_metrics = [] # vs Ori
quality_metrics_recon = [] # vs Recon
quality_metrics_codec = [] # Recon vs Ori (Codec Only)
TARGET_CKPT = args.checkpoint



# Create audio output directory
output_sample_dir = Path(args.output_dir)
output_sample_dir.mkdir(parents=True, exist_ok=True)
has_saved_sample = False

for attack_name, attack_fn in test_attacks.items():
    print(f"\n>>> Testing attack: {attack_name} | Model: {os.path.basename(TARGET_CKPT)} ...")
    results_wm = []      # Watermarked results
    results_clean_ori = [] # Original clean audio
    results_clean_recon = [] # Reconstructed clean audio
    
    # Retrieve global solver instance in advance
    solver_instance = get_solver(TARGET_CKPT, device=device)
    solver_instance.model.eval()
    
    for fpath in tqdm(test_files, desc=f"{attack_name[:15]}"):
        try:
            wav, sr = torchaudio.load(fpath)
            wav = wav.to(device)
            
            # Prepare 16k original audio as baseline
            # 1. Convert to mono immediately (aligned with dataset/audio_dataset.py)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 2. Resample (aligned with dataset/audio_dataset.py using torchaudio)
            if sr != 16000:
                wav_16k = torchaudio.functional.resample(wav, sr, 16000)
            else:
                wav_16k = wav
            
            # 3. Normalization: Do not manually normalize as dataset/audio_dataset.py does not do this
            # Keep original amplitude (fully aligned with training phase)
            
            # Crucial: Crop to fixed length during training (segment_size=48000, downsample_rate=320)
            SEGMENT_SIZE = 48000
            DOWNSAMPLE_RATE = 320
            if wav_16k.shape[-1] > SEGMENT_SIZE:
                wav_16k = wav_16k[..., :SEGMENT_SIZE]
            
            # Ensure length is a multiple of downsample_rate
            trim_len = wav_16k.shape[-1] - (wav_16k.shape[-1] % DOWNSAMPLE_RATE)
            wav_16k = wav_16k[..., :trim_len]
            
            if wav_16k.shape[-1] < DOWNSAMPLE_RATE:
                print(f"Skipping too-short file: {fpath}")
                continue
            
            test_message = "".join(random.choice("01") for _ in range(16))
            
            with torch.no_grad():
                # --- 1. Generation Phase (same as train.py) ---
                # Convert message to tensor
                msg_tensor = torch.tensor([int(b) for b in test_message], dtype=torch.int64).to(device).unsqueeze(0)
                
                # Explicitly prepare cln_input [1, 1, T]
                if wav_16k.dim() == 1:
                    cln_input = wav_16k.unsqueeze(0).unsqueeze(0)
                elif wav_16k.dim() == 2:
                    cln_input = wav_16k.unsqueeze(0)
                else:
                    cln_input = wav_16k
 
                # Generate directly using model
                # WatermarkModel returns {"recon": ..., "recon_wm": ...}
                res_gen = solver_instance.model(cln_input, message=msg_tensor)
                recon_audio = res_gen["recon"]
                wm_audio = res_gen["recon_wm"]
 
                # Save an example audio (only executed once in the first Clean test)
                if not has_saved_sample and attack_name == "Clean (Identity)":
                    print(f"Saving example audio to {output_sample_dir} ...")
                    torchaudio.save(output_sample_dir / "original.wav", cln_input.squeeze(0).cpu(), 16000)
                    torchaudio.save(output_sample_dir / "reconstructed.wav", recon_audio.squeeze(0).cpu(), 16000)
                    torchaudio.save(output_sample_dir / "watermarked.wav", wm_audio.squeeze(0).cpu(), 16000)
                    has_saved_sample = True
 
                # B. Compute transparency quality (ensure consistent sample rate)
                if attack_name == "Clean (Identity)":
                    # 1. Compare to Original (Ref Ori)
                    p_score, s_score, snr = compute_quality(wav_16k, wm_audio, 16000)
                    quality_metrics.append((p_score, s_score, snr))
                    # 2. Compare to Reconstructed (Ref Recon) - reflects watermark quality purely
                    p_score_r, s_score_r, snr_r = compute_quality(recon_audio, wm_audio, 16000)
                    quality_metrics_recon.append((p_score_r, s_score_r, snr_r))
                    # 3. Compare to Codec (Recon vs Ori) - reflects codec inherent loss
                    p_score_c, s_score_c, snr_c = compute_quality(wav_16k, recon_audio, 16000)
                    quality_metrics_codec.append((p_score_c, s_score_c, snr_c))
 
                # --- 2. Attack Phase ---
                attacked_wm = attack_fn(wm_audio, 16000)
                attacked_ori = attack_fn(cln_input, 16000)
                attacked_recon = attack_fn(recon_audio, 16000)
 
                # --- 3. Detection Phase (same as detect_watermark logic in train.py) ---
                def get_stats(audio):
                    # Extract features and detect
                    feats = solver_instance.model.st_model.forward_feature(audio)
                    prob, msg_out_tensor, _ = solver_instance.model.detector.detect_watermark(feats)
                    msg_out = msg_out_tensor.squeeze(0).cpu().numpy().tolist()
                    # Compute bit accuracy (compared to original message)
                    matches = sum(c1 == int(c2) for c1, c2 in zip(msg_out, test_message))
                    acc = matches / len(test_message)
                    return float(prob), acc
 
                prob_wm, acc_wm = get_stats(attacked_wm)
                prob_ori, acc_ori = get_stats(attacked_ori)
                prob_recon, acc_recon = get_stats(attacked_recon)
            
            # --- 4. Collect results ---
            results_wm.append({"bit_acc": acc_wm, "prob": prob_wm})
            results_clean_ori.append({"bit_acc": acc_ori, "prob": prob_ori})
            results_clean_recon.append({"bit_acc": acc_recon, "prob": prob_recon})
            
        except Exception as e:
            print(f"Error processing {fpath}: {str(e)}")
            continue
        finally:
            # Release GPU memory to prevent buildup during long sequence testing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
    if results_wm:
        # Aggregate detection accuracy (directly use probability values to compute standard ACC, no threshold)
        tp = sum(r["prob"] for r in results_wm)
        tn = sum(1.0 - r["prob"] for r in results_clean_ori)
        detect_acc = (tp + tn) / (len(results_wm) + len(results_clean_ori))
 
        avg_acc_wm = sum(r["bit_acc"] for r in results_wm) / len(results_wm)
        avg_prob_wm = sum(r["prob"] for r in results_wm) / len(results_wm)
        
        avg_acc_ori = sum(r["bit_acc"] for r in results_clean_ori) / len(results_clean_ori)
        avg_prob_ori = sum(r["prob"] for r in results_clean_ori) / len(results_clean_ori)
        
        avg_acc_recon = sum(r["bit_acc"] for r in results_clean_recon) / len(results_clean_recon)
        avg_prob_recon = sum(r["prob"] for r in results_clean_recon) / len(results_clean_recon)
        
        summary_report[attack_name] = {
            "wm_bit_acc": avg_acc_wm,
            "wm_prob": avg_prob_wm,
            "ori_bit_acc": avg_acc_ori,
            "ori_prob": avg_prob_ori,
            "recon_bit_acc": avg_acc_recon,
            "recon_prob": avg_prob_recon,
            "detect_acc": detect_acc
        }
 
# 6.2 Classified and sorted output
print(f"\n=============================================================================================================================")
print(f"Test model: {TARGET_CKPT}")
print(f"{'='*135}")
print(f"{'Attack Type':<22} | {'Detect ACC':<12} | {'WM Bit Acc':<12} | {'WM Prob':<10} | {'Ori B.Acc':<10} | {'Ori Prob':<10} | {'Rec B.Acc':<10} | {'Rec Prob':<10}")
print(f"{'-'*135}")
 
# Define categories
codec_keywords = ["Encodec", "DAC", "WavTokenizer"]
codec_names = []
other_names = []
clean_name = "Clean (Identity)"
 
for name in summary_report.keys():
    if name == clean_name:
        continue
    if any(k in name for k in codec_keywords):
        codec_names.append(name)
    else:
        other_names.append(name)
 
# Final sorted list: Clean -> Other general attacks (sorted) -> Codec attacks (sorted)
first_part = [clean_name] if clean_name in summary_report else []
first_part += sorted(other_names)
 
for name in first_part:
    stats = summary_report[name]
    wm_acc = stats["wm_bit_acc"]
    wm_prob = stats["wm_prob"]
    ori_acc = stats["ori_bit_acc"]
    ori_prob = stats["ori_prob"]
    recon_acc = stats["recon_bit_acc"]
    recon_prob = stats["recon_prob"]
    d_acc = stats["detect_acc"]
    
    print(f"{name:<22} | {d_acc:<12.4f} | {wm_acc:<12.4f} | {wm_prob:<10.4f} | {ori_acc:<10.4f} | {ori_prob:<10.4f} | {recon_acc:<10.4f} | {recon_prob:<10.4f}")
 
print("-" * 135)
 
for name in sorted(codec_names):
    stats = summary_report[name]
    wm_acc = stats["wm_bit_acc"]
    wm_prob = stats["wm_prob"]
    ori_acc = stats["ori_bit_acc"]
    ori_prob = stats["ori_prob"]
    recon_acc = stats["recon_bit_acc"]
    recon_prob = stats["recon_prob"]
    d_acc = stats["detect_acc"]
    
    display_name = name + " *" if any(k in name for k in ["DAC", "WavTokenizer"]) else name
    print(f"{display_name:<22} | {d_acc:<12.4f} | {wm_acc:<12.4f} | {wm_prob:<10.4f} | {ori_acc:<10.4f} | {ori_prob:<10.4f} | {recon_acc:<10.4f} | {recon_prob:<10.4f}")
print("-" * 125)
if quality_metrics:
    avg_pesq = sum(m[0] for m in quality_metrics) / len(quality_metrics)
    avg_stoi = sum(m[1] for m in quality_metrics) / len(quality_metrics)
    avg_snr = sum(m[2] for m in quality_metrics) / len(quality_metrics)
    print(f"Transparency (vs Original):")
    print(f"  - Avg PESQ (WB): {avg_pesq:.4f}")
    print(f"  - Avg STOI:      {avg_stoi:.4f}")
    print(f"  - Avg SI-SNR:    {avg_snr:.4f} dB")
 
if quality_metrics_recon:
    avg_pesq_r = sum(m[0] for m in quality_metrics_recon) / len(quality_metrics_recon)
    avg_stoi_r = sum(m[1] for m in quality_metrics_recon) / len(quality_metrics_recon)
    avg_snr_r = sum(m[2] for m in quality_metrics_recon) / len(quality_metrics_recon)
    print(f"Transparency (vs Reconstructed - Ref Recon):")
    print(f"  - Avg PESQ (WB): {avg_pesq_r:.4f}")
    print(f"  - Avg STOI:      {avg_stoi_r:.4f}")
    print(f"  - Avg SI-SNR:    {avg_snr_r:.4f} dB")
 
if quality_metrics_codec:
    avg_pesq_c = sum(m[0] for m in quality_metrics_codec) / len(quality_metrics_codec)
    avg_stoi_c = sum(m[1] for m in quality_metrics_codec) / len(quality_metrics_codec)
    avg_snr_c = sum(m[2] for m in quality_metrics_codec) / len(quality_metrics_codec)
    print(f"Codec Only (Reconstructed vs Original):")
    print(f"  - Avg PESQ (WB): {avg_pesq_c:.4f}")
    print(f"  - Avg STOI:      {avg_stoi_c:.4f}")
    print(f"  - Avg SI-SNR:    {avg_snr_c:.4f} dB")
print("="*105)
