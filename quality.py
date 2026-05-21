import torch
from pesq import pesq
from pystoi import stoi

def compute_quality(ref_tensor, deg_tensor, sr=16000):
    """Utility to compute PESQ, STOI and SI-SNR from tensors."""
    ref = ref_tensor.detach().cpu().squeeze()
    deg = deg_tensor.detach().cpu().squeeze()
    
    # SI-SNR (Torch based)
    with torch.no_grad():
        r = ref_tensor.detach().reshape(-1)
        d = deg_tensor.detach().reshape(-1)
        min_len = min(len(r), len(d))
        r = r[:min_len]
        d = d[:min_len]
        
        # Scale-invariant projection
        dot = torch.sum(r * d)
        ref_pow = torch.sum(r ** 2) + 1e-8
        target = (dot / ref_pow) * r
        noise = d - target
        
        target_pow = torch.sum(target ** 2) + 1e-8
        noise_pow = torch.sum(noise ** 2) + 1e-8
        
        snr_score = (10 * torch.log10(target_pow / noise_pow)).item()
    
    # PESQ/STOI (Numpy based)
    ref_np = ref.numpy()
    deg_np = deg.numpy()
    min_len_np = min(len(ref_np), len(deg_np))
    ref_np = ref_np[:min_len_np]
    deg_np = deg_np[:min_len_np]
    
    try:
        pesq_score = pesq(sr, ref_np, deg_np, 'wb')
    except Exception:
        pesq_score = 0.0
    try:
        stoi_score = stoi(ref_np, deg_np, sr, extended=False)
    except Exception:
        stoi_score = 0.0
        
    return pesq_score, stoi_score, snr_score
