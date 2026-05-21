import torch
import random
from .effects import AudioEffects, EncodecAttack, DACAttack, WavTokenizerAttack
import os

def apply_encodec(wav, sr, bw):
    """Wrapper around EncodecAttack.encodec_compress for validation and external scripts."""
    return EncodecAttack.encodec_compress(wav, bandwidth=bw, sample_rate=sr)

def apply_dac(wav, sr, target_sr=16000):
    """Wrapper around DACAttack.dac_compress for validation and external scripts."""
    model_type = "16khz" if target_sr == 16000 else "24khz"
    return DACAttack.dac_compress(wav, model_type=model_type, sample_rate=sr)

def apply_wavtokenizer(wav, sr):
    """Wrapper around WavTokenizerAttack.wavtokenizer_compress for validation and external scripts."""
    return WavTokenizerAttack.wavtokenizer_compress(wav, sample_rate=sr)

def apply_masking(audio, orig_audio, sample_rate, mask_prob=0.2):
    """
    Simulates Voice Conversion by masking random segments with original (unwatermarked) audio.
    Returns: (augmented_audio, vad_labels)
    """
    B, C, T = audio.shape
    downsampling_ratios = 320
    n_frames = T // downsampling_ratios
    
    # Initialize labels as 1.0
    vad_labels = torch.ones(B, n_frames, device=audio.device)
    
    # Generate random mask
    mask = torch.rand(B, n_frames, device=audio.device) < mask_prob
    vad_labels[mask] = 0.0
    
    # Prepare audio mask
    mask_expanded = mask.view(B, 1, n_frames, 1).expand(-1, -1, -1, downsampling_ratios)
    mask_expanded = mask_expanded.reshape(B, 1, n_frames * downsampling_ratios)
    
    if mask_expanded.shape[-1] < T:
        pad_len = T - mask_expanded.shape[-1]
        mask_pad = torch.zeros(B, 1, pad_len, dtype=torch.bool, device=mask.device)
        mask_expanded = torch.cat([mask_expanded, mask_pad], dim=-1)
    
    # Apply masking: replace watermarked segments with original unwatermarked audio
    augmented = audio.clone()
    # If orig_audio is provided, use it; otherwise use zeros (silence)
    fill_source = orig_audio if orig_audio is not None else torch.zeros_like(audio)
    
    # Ensure fill_source matches length
    if fill_source.shape[-1] < T:
        fill_source = torch.nn.functional.pad(fill_source, (0, T - fill_source.shape[-1]))
    else:
        fill_source = fill_source[..., :T]
        
    augmented[mask_expanded] = fill_source[mask_expanded]
    
    return augmented, vad_labels

# Unified Registry: All functions return (augmented_audio, vad_labels_or_None)
ATTACK_REGISTRY = {
    "identity": lambda audio, sr: (audio, None),
    "vc_masking": lambda audio, sr: (audio, "MASK"), # Special token to trigger masking
    "encodec_3kbps": lambda audio, sr: (apply_encodec(audio, sr, 3.0), None),
    "encodec_6kbps": lambda audio, sr: (apply_encodec(audio, sr, 6.0), None),
    "encodec_12kbps": lambda audio, sr: (apply_encodec(audio, sr, 12.0), None),
    "random_noise": lambda audio, sr: (AudioEffects.random_noise(audio, noise_std=0.001), None),
    "pink_noise": lambda audio, sr: (AudioEffects.pink_noise(audio, noise_std=0.01), None),
    "lowpass_filter": lambda audio, sr: (AudioEffects.lowpass_filter(audio, cutoff_freq=5000, sample_rate=sr), None),
    "highpass_filter": lambda audio, sr: (AudioEffects.highpass_filter(audio, cutoff_freq=500, sample_rate=sr), None),
    "bandpass_filter": lambda audio, sr: (AudioEffects.bandpass_filter(audio, cutoff_freq_low=300, cutoff_freq_high=8000, sample_rate=sr), None),
    "echo": lambda audio, sr: (AudioEffects.echo(audio, volume_range=(0.1, 0.5), duration_range=(0.1, 0.5), sample_rate=sr), None),
    "smooth": lambda audio, sr: (AudioEffects.smooth(audio, window_size_range=(2, 10)), None),
    "boost_audio": lambda audio, sr: (AudioEffects.boost_audio(audio, amount=10), None),
    "duck_audio": lambda audio, sr: (AudioEffects.duck_audio(audio, amount=10), None),
    "shush": lambda audio, sr: (AudioEffects.shush(audio), None),
    "resample": lambda audio, sr: (AudioEffects.updownresample(audio, sample_rate=sr, intermediate_freq=32000), None),
    "speed": lambda audio, sr: (AudioEffects.speed(audio, speed_range=(0.9, 1.1), sample_rate=sr), None),
    "dac_16k": lambda audio, sr: (apply_dac(audio, sr, 16000), None),
    "dac_24k": lambda audio, sr: (apply_dac(audio, sr, 24000), None),
    "wavtokenizer": lambda audio, sr: (apply_wavtokenizer(audio, sr), None),
}

def attack_augmentation(audio, sample_rate, attack_name=None, orig_audio=None):
    """
    Unified entry point for all augmentations. 
    If attack_name is None, it samples randomly based on predefined weights.
    Returns: (augmented_audio, vad_labels, attack_name)
    """
    B, C, T = audio.shape
    downsampling_ratios = 320
    n_frames = T // downsampling_ratios
    
    # Random sampling with weights if no specific attack is requested
    if attack_name is None:
        attack_keys = list(ATTACK_REGISTRY.keys())
        # Exclude heavy codec attacks (DAC, WavTokenizer) during training phase, only manually specify them during validation
        heavy_codecs = ["dac_16k", "dac_24k", "wavtokenizer"]
        
        weights = []
        for k in attack_keys:
            if k in heavy_codecs:
                weights.append(0) # Do not randomly select them during training
            elif k == "identity":
                weights.append(10)
            elif k == "vc_masking":
                weights.append(5)
            else:
                weights.append(1)
                
        attack_name = random.choices(attack_keys, weights=weights, k=1)[0]
    
    if attack_name not in ATTACK_REGISTRY:
        attack_name = "identity"
        
    atk_fn = ATTACK_REGISTRY[attack_name]
    augmented, vad_labels = atk_fn(audio, sample_rate)
    
    # Handle the special masking case
    if vad_labels == "MASK":
        # First apply a random non-masking attack to make it even harder
        heavy_codecs = ["dac_16k", "dac_24k", "wavtokenizer"]
        sub_attacks = [k for k in ATTACK_REGISTRY.keys() if k not in ["identity", "vc_masking"] + heavy_codecs]
        sub_atk_name = random.choice(sub_attacks)
        augmented, _ = ATTACK_REGISTRY[sub_atk_name](audio, sample_rate)
        # Then apply masking
        augmented, vad_labels = apply_masking(augmented, orig_audio, sample_rate)
    
    # Default vad_labels for standard attacks
    if vad_labels is None:
        vad_labels = torch.ones(B, n_frames, device=audio.device)
        
    # Consistency checks (ensure length hasn't changed)
    if augmented.shape[-1] != T:
        if augmented.shape[-1] > T:
            augmented = augmented[..., :T]
        else:
            augmented = torch.nn.functional.pad(augmented, (0, T - augmented.shape[-1]))
            
    return torch.clamp(augmented, -1.0, 1.0), vad_labels, attack_name
