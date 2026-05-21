"""Discriminator architectures used by NeuMark training."""

from .discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)

__all__ = [
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "MultiScaleSTFTDiscriminator",
]
