"""Compatibility layer that provides the Organism enum expected by AlphaGenome PyTorch."""

from __future__ import annotations

from enum import Enum


class Organism(Enum):
    HOMO_SAPIENS = 0
    MUS_MUSCULUS = 1
