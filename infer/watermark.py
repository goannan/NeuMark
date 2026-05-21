"""Watermark inference model and checkpoint loading for NeuMark validation."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models import WMDetector, WMEmbedder
from STmodels.model import SpeechTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WatermarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        config_path = PROJECT_ROOT / "STmodels" / "pretrained_model" / "speechtokenizer_hubert_avg_config.json"
        ckpt_path = PROJECT_ROOT / "STmodels" / "pretrained_model" / "SpeechTokenizer.pt"
        self.st_model = SpeechTokenizer.load_from_checkpoint(str(config_path), str(ckpt_path))
        self.msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4)
        self.detector = WMDetector(1024, 16, nchunk_size=4)

    def detect_watermark(self, x: torch.Tensor, return_logits=False) -> Tuple[float, torch.Tensor]:
        embedding = self.st_model.forward_feature(x)
        if return_logits:
            return self.detector(embedding)
        return self.detector.detect_watermark(embedding)

    def forward(
        self,
        speech_input: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon, recon_wm, acoustic, acoustic_wm = self.st_model(
            speech_input,
            msg_processor=self.msg_processor,
            message=message,
        )
        wav_length = min(speech_input.size(-1), recon_wm.size(-1))
        return {
            "recon": recon[..., :wav_length],
            "recon_wm": recon_wm[..., :wav_length],
        }


class WatermarkSolver:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WatermarkModel().to(self.device)
        self.epoch = -1

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def load_model(self, checkpoint, strict=False):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading model weights from {checkpoint_path}...")
        checkpoint_obj = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        model_state_dict = self._extract_model_state_dict(checkpoint_obj)
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

        if not strict:
            current_state = self.model.state_dict()
            model_state_dict = {
                k: v for k, v in model_state_dict.items()
                if k in current_state and current_state[k].shape == v.shape
            }

        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.epoch = checkpoint_obj.get("epoch", -1) if isinstance(checkpoint_obj, dict) else -1
        print("Model state dict loaded successfully.")
        print("   Missing keys:", missing)
        print("   Unexpected keys:", unexpected)

    def _extract_model_state_dict(self, checkpoint):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "msg_processor" in checkpoint and "detector" in checkpoint:
            return {
                **{f"msg_processor.{k}": v for k, v in checkpoint["msg_processor"].items()},
                **{f"detector.{k}": v for k, v in checkpoint["detector"].items()},
            }

        if "embedder" in checkpoint and "detector" in checkpoint:
            return {
                **{f"msg_processor.{k}": v for k, v in checkpoint["embedder"].items()},
                **{f"detector.{k}": v for k, v in checkpoint["detector"].items()},
            }

        if "embedder" in checkpoint and "detectors" in checkpoint:
            print("Warning: checkpoint has 'detectors', which are discriminators in older runs; loading embedder only.")
            return {f"msg_processor.{k}": v for k, v in checkpoint["embedder"].items()}

        if "generator" in checkpoint and "watermark_encoder" in checkpoint and "watermark_decoder" in checkpoint:
            state_dict = {}
            state_dict.update({f"msg_processor.{k}": v for k, v in checkpoint["watermark_encoder"].items()})
            state_dict.update({f"detector.{k}": v for k, v in checkpoint["watermark_decoder"].items()})
            state_dict.update({f"generator.{k}": v for k, v in checkpoint.get("generator", {}).items()})
            return state_dict

        keys = list(checkpoint.keys())
        has_speech_tokenizer = (
            any(k.startswith("encoder.") for k in keys)
            and any(k.startswith("decoder.") for k in keys)
            and any(k.startswith("quantizer.") for k in keys)
        )
        has_watermark = any(k.startswith("msg_processor.") for k in keys) or any(k.startswith("detector.") for k in keys)
        if has_speech_tokenizer and not has_watermark:
            return {f"st_model.{k}": v for k, v in checkpoint.items()}

        return checkpoint


_solver_cache = {}


def get_solver(checkpoint, device=None):
    checkpoint_path = Path(checkpoint)
    cache_key = str(checkpoint_path if checkpoint_path.is_absolute() else PROJECT_ROOT / checkpoint_path)
    if cache_key not in _solver_cache:
        solver = WatermarkSolver(device=device)
        solver.load_model(checkpoint)
        _solver_cache[cache_key] = solver
    elif device is not None:
        _solver_cache[cache_key].to(device)
    return _solver_cache[cache_key]
