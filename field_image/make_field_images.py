"""Write the electron-field pictures to images/."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from field_image import A_COLOR, H_COLOR, field_figure, save_field

OUT = Path(__file__).parent / "images"

VARIANTS = {
    "electron_field_red": dict(color=A_COLOR),
    "electron_field_red_labelled": dict(color=A_COLOR, label="ELECTRON"),
    "electron_field_blue": dict(color=H_COLOR),
    "electron_field_blue_labelled": dict(color=H_COLOR, label="ELECTRON"),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, kwargs in VARIANTS.items():
        fig = field_figure(**kwargs)
        save_field(fig, str(OUT / name))
        print("wrote", name)
