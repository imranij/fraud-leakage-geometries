# -*- coding: utf-8 -*-
"""Shared palette and rcParams for the three schematic figures.

Reused across fig1_geometries, fig2_adaptivity, fig3_platform so the set
reads as one system. Colours are chosen to remain distinguishable when
printed in grayscale (they differ in lightness, not only in hue).
"""

import matplotlib.pyplot as plt

# Structural / neutral (dark, near-black navy) -- boxes, borders, generic text
INK = "#1b2a4a"
# Neutral mid tone -- secondary boxes, connective arrows
SLATE = "#5b6b7c"
# Fraud channel -- warm, adversarial (darker in grayscale ~ mid-dark gray)
FRAUD = "#b1473e"
FRAUD_FILL = "#f3ded9"
# Leakage channel -- cool, process-oriented (darker in grayscale ~ dark gray)
LEAK = "#1f7a72"
LEAK_FILL = "#d9ece8"
# Generic light fills
BOX_FILL = "#eef2f5"
SUPPORT_FILL = "#dbe6ef"
BG = "#ffffff"


def set_rcparams(base_size=8.5):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
