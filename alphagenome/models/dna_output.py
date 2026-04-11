"""Compatibility layer that provides the OutputType enum expected by AlphaGenome PyTorch."""

from __future__ import annotations

from enum import Enum


class OutputType(Enum):
    ATAC = 1
    CAGE = 2
    DNASE = 3
    RNA_SEQ = 4
    PROCAP = 5
    CHIP_HISTONE = 6
    CHIP_TF = 7
    CONTACT_MAPS = 8
    SPLICE_SITES = 9
    SPLICE_SITE_USAGE = 10
    SPLICE_JUNCTIONS = 11
