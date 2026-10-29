"""Write the oscillator icons (bare curve, transparent background) to icons/."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
from oscillator_plots import icon_figure, ring_icon_figure, save_icon, save_ring_icon

OUT = Path(__file__).parent / "icons"

VARIANTS = {
    "oscillator_icon_n1": dict(n=1),
    "oscillator_icon_n2": dict(n=2),
    "oscillator_icon_n3": dict(n=3),
    "oscillator_icon_superposition_0_1": dict(coefficients={0: 1.0, 1: 1.0}),
}

# The same wave wrapped round a ring: m_l wiggles, for the orbital angular momentum slide. The rotated copy is
# turned through 2 pi / (4 m_l), a quarter of one wiggle, so the before/after pair differs visibly.
_QUARTER_M5 = 2 * np.pi / (4 * 5)
RING_VARIANTS = {
    "ring_icon_m3": dict(m_l=3, amplitude=0.15),
    "ring_icon_m5": dict(m_l=5, amplitude=0.12),
    "ring_icon_m6": dict(m_l=6, amplitude=0.10),
    "ring_icon_m5_rotated": dict(m_l=5, amplitude=0.12, rotation=_QUARTER_M5),
    "ring_icon_m5_reference": dict(m_l=5, amplitude=0.12, reference=True),
    "ring_icon_m5_glow": dict(m_l=5, amplitude=0.12, glow=True),
    "ring_icon_m5_rotated_glow": dict(m_l=5, amplitude=0.12, rotation=_QUARTER_M5, glow=True),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, kwargs in VARIANTS.items():
        fig = icon_figure(**kwargs)
        save_icon(fig, str(OUT / name))
        print("wrote", name)
    for name, kwargs in RING_VARIANTS.items():
        fig = ring_icon_figure(**kwargs)
        save_ring_icon(fig, str(OUT / name))
        print("wrote", name)
