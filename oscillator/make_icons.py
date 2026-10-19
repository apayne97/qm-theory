"""Write the oscillator icons (bare curve, transparent background) to icons/."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from oscillator_plots import icon_figure, save_icon

OUT = Path(__file__).parent / "icons"

VARIANTS = {
    "oscillator_icon_n1": dict(n=1),
    "oscillator_icon_n2": dict(n=2),
    "oscillator_icon_n3": dict(n=3),
    "oscillator_icon_superposition_0_1": dict(coefficients={0: 1.0, 1: 1.0}),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, kwargs in VARIANTS.items():
        fig = icon_figure(**kwargs)
        save_icon(fig, str(OUT / name))
        print("wrote", name)
