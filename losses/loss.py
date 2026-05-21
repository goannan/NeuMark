import torch
import torchaudio
import matplotlib.pylab as plt
import torch.nn as nn
import torch.nn.functional as F

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    n_fft = int(n_fft)
    num_mels = int(num_mels)
    sample_rate = int(sample_rate)
    hop_size = int(hop_size)
    win_size = int(win_size)
    fmin = float(fmin)
    fmax = float(fmax)

    global mel_basis, hann_window
    key = f"{fmax}_{win_size}_{y.device}"
    if key not in mel_basis:
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=sample_rate,
            n_stft=n_fft//2 + 1,
            f_min=fmin,
            f_max=fmax,
            norm='slaney',
            mel_scale='htk'
        )
        mel_basis[key] = mel_transform.fb.float().T.to(y.device)
    
    if key not in hann_window:
        hann_window[key] = torch.hann_window(win_size).to(y.device)


    # Pad waveform
    y = torch.nn.functional.pad(y.unsqueeze(1),
                                (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)),
                                mode='reflect').squeeze(1)

    spec_complex = torch.stft(
        y, n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[key],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True  
    )

    spec = torch.view_as_real(spec_complex)  # shape: (..., 2)， [real, imag]

    spec = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2 + 1e-9)

    # trans to Mel
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def mel_loss(x, x_hat, **kwargs):
    x_mel = mel_spectrogram(x.squeeze(1), **kwargs)
    x_hat_mel = mel_spectrogram(x_hat.squeeze(1), **kwargs)

    length = min(x_mel.size(2), x_hat_mel.size(2))

    return torch.nn.functional.l1_loss(x_mel[..., :length], x_hat_mel[..., :length])


def multi_scale_mel_loss(x, x_hat, scales, num_mels, sample_rate, fmin, fmax, device=None, weights=None):
    """
    Multi-scale Mel Spectrogram L1 loss.

    Parameters
    ----------
    x, x_hat : torch.Tensor
        Waveforms with shape (B, 1, T)
    scales : list[int]
        List of n_fft values to use (e.g., [1024, 512, 256])
    num_mels : int
    sample_rate, fmin, fmax : as usual
    device : torch.device or None
    weights : list[float] or None
        Per-scale weights. If None, equal weights are used.

    Returns
    -------
    loss : torch.Tensor
    """
    if device is None:
        device = x.device

    if weights is None:
        weights = [1.0] * len(scales)

    total = 0.0
    for s, w in zip(scales, weights):
        mel_kwargs = dict(n_fft=s, num_mels=num_mels, sample_rate=sample_rate,
                          hop_size=s//4, win_size=s, fmin=fmin, fmax=fmax)
        l = mel_loss(x, x_hat, **mel_kwargs)
        total += w * l

    return total

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def decoding_loss(w_hat: torch.Tensor, w_true: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for watermark decoding.

    Parameters
    ----------
    w_hat : torch.Tensor
        Predicted watermark logits, shape (batch, n_bits, n_classes)
    w_true : torch.Tensor
        Ground truth watermark indices, shape (batch, n_bits)

    Returns
    -------
    loss : torch.Tensor
        Scalar loss
    """
    # Use nn.CrossEntropyLoss needs to change w_hat.shape to (batch*n_bits, n_classes)
    # w_true -> (batch*n_bits)
    batch, n_bits, n_classes = w_hat.shape
    w_hat_flat = w_hat.view(batch * n_bits, n_classes)
    w_true_flat = w_true.view(batch * n_bits)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(w_hat_flat, w_true_flat)
    return loss


def dual_threshold_vad(audio: torch.Tensor, n_fft: int, hop_size: int, win_size: int,
                       low_threshold: float = 0.01, high_threshold: float = 0.03) -> torch.Tensor:
    """
    Simple dual-threshold VAD applied on frame energies with hysteresis.

    Parameters
    ----------
    audio : torch.Tensor
        Waveform tensor shape (B, 1, T) or (B, T)
    n_fft, hop_size, win_size : int
    low_threshold, high_threshold : float
        Energy thresholds (in linear power domain). high_threshold > low_threshold.

    Returns
    -------
    vad_labels : torch.Tensor
        Binary labels per frame with shape (B, n_frames), dtype=torch.float32
    """
    # Ensure shape (B, T)
    if audio.dim() == 3:
        audio = audio.squeeze(1)

    # compute frame energies via STFT magnitude mean across bins
    window = torch.hann_window(win_size, device=audio.device)
    spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size,
                      window=window, center=False, return_complex=True)
    mag = torch.abs(spec)  # (B, freq_bins, n_frames)
    energy = mag.pow(2).mean(dim=1)  # (B, n_frames)

    # Normalize energy per-batch to make thresholds relative
    eps = 1e-9
    energy = energy / (energy.mean(dim=1, keepdim=True) + eps)

    B, N = energy.shape
    vad = torch.zeros_like(energy, dtype=torch.float32)

    for b in range(B):
        state = 0
        for t in range(N):
            e = energy[b, t].item()
            if state == 0:
                if e >= high_threshold:
                    state = 1
            else:
                if e < low_threshold:
                    state = 0
            vad[b, t] = float(state)

    return vad

def bits_to_chunks(bits: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
    """
    Convert bit-level watermark (0/1) into chunk-level integers.

    Parameters
    ----------
    bits : torch.Tensor
        Shape (batch, n_bits), values in {0,1}
    chunk_size : int
        Number of bits per chunk (default: 4)

    Returns
    -------
    chunks : torch.Tensor
        Shape (batch, n_chunks), each value in [0, 2^chunk_size - 1]
    """
    batch, n_bits = bits.shape
    assert n_bits % chunk_size == 0, f"n_bits={n_bits} must be divisible by chunk_size={chunk_size}"

    n_chunks = n_bits // chunk_size
    bits_reshaped = bits.view(batch, n_chunks, chunk_size)  # [batch, n_chunks, 4]

    # Convert each chunk of bits into integer using LSB-first (matching WMEmbedder.forward)
    # e.g., [1,0,1,1] → 1*1 + 0*2 + 1*4 + 1*8 = 13
    weights = 2 ** torch.arange(chunk_size, device=bits.device)  # [1,2,4,8]
    chunks = (bits_reshaped * weights).sum(dim=-1).long()  # [batch, n_chunks]

    return chunks


def vad_based_loss(vad_pred: torch.Tensor, vad_label: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """
    VAD-based Loss for watermarked speech.

    Parameters
    ----------
    vad_pred : torch.Tensor
        Model's predicted values.
        If from_logits=True, vad_pred is raw logits. Shape: (batch, timesteps)
        If from_logits=False, vad_pred should already be probabilities in [0,1].
    vad_label : torch.Tensor
        Binary label for each frame. 1 = speech+watermark, 0 = silent/masked/replaced.
        Shape: (batch, timesteps)
    from_logits : bool
        Whether vad_pred is raw logits (default True).

    Returns
    -------
    loss : torch.Tensor
        Scalar BCE loss
    """
    if from_logits:
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(vad_pred, vad_label.float())
    else:
        # If already sigmoid probabilities
        vad_pred = torch.clamp(vad_pred, 1e-6, 1 - 1e-6)
        criterion = nn.BCELoss()
        loss = criterion(vad_pred, vad_label.float())

    return loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def adversarial_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l

    return loss

def cos_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity loss.

    Args:
        x: Tensor of shape (B, C, T) or (B, T) - predicted feature or audio
        y: Tensor of same shape as x - target feature or audio
        eps: small value to avoid division by zero

    Returns:
        scalar tensor - cosine loss
    """
    # flatten channel/time dimension if necessary
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)

    # cosine similarity: (B,)
    cos_sim = F.cosine_similarity(x_flat, y_flat, dim=1, eps=eps)

    # loss = 1 - cosine similarity (more close to 1 means more similar)
    loss = 1 - cos_sim

    # return mean over batch
    return loss.mean()

def adversarial_loss_d(real_preds, fake_preds):
    """
    discriminator loss
    real_preds: D(x) (the output of real audio input to discriminator)
    fake_preds: D(x_hat) (the output of discriminator when fed watermarked audio)
    """
    loss_real = torch.mean((real_preds - 1) ** 2)  # real audio hopes to output 1
    loss_fake = torch.mean(fake_preds ** 2)        # fake audio hopes to output 0
    return loss_real + loss_fake


def adversarial_loss_g(fake_preds):
    """
    generator adversarial loss
    fake_preds: D(x_hat) output (the output of discriminator when fed watermarked audio)
    """
    return torch.mean((fake_preds - 1) ** 2)  # generator hopes discriminator outputs 1 for its fake audio