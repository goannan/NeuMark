"""Small convolution wrappers used by the adversarial discriminators."""

import typing as tp

from torch import nn
from torch.nn.utils import spectral_norm, weight_norm


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm", "spectral_norm", "time_group_norm"])


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "spectral_norm":
        return spectral_norm(module)
    return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs):
    assert norm in CONV_NORMALIZATIONS
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm does not support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


class NormConv1d(nn.Module):
    """Conv1d plus optional weight/spectral/group normalization."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConv2d(nn.Module):
    """Conv2d plus optional weight/spectral/group normalization."""

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        return self.norm(self.conv(x))
