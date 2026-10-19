"""Regenerate the figures of the knowledge-base quantum-mechanics chapter from the notebook plotting modules.

Run this after changing `qmstyle.py` or a plotting function, then rebuild the chapter.
"""

import math
import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
for folder in ("plane_waves", "oscillator", "quantization", "spinors"):
    sys.path.insert(0, str(HERE / folder))

import belt_plots  # noqa: E402
import quantization_plots  # noqa: E402
import wave_plots  # noqa: E402

DEFAULT_OUT = HERE / "chapter" / "figures"

FIGURES = {
    "wave-phase.png": lambda: wave_plots.phase_figure(),
    "wave-time-features.png": lambda: wave_plots.time_features_figure(),
    "wave-space-features.png": lambda: wave_plots.space_features_figure(),
    "wave-travelling-features.png": lambda: wave_plots.travelling_features_figure(),
    "wave-surface.png": lambda: wave_plots.wave_surface_figure(1.5, 1.5, 2.0),
    "two-hole-schematic.png": lambda: wave_plots.interference_schematic(),
    "two-hole-interference.png": lambda: wave_plots.interference_figure(1.25, 1.0, 4.0),
    "oscillator-energy.png": lambda: wave_plots.oscillator_energy_figure(),
    "packet-times.png": lambda: wave_plots.packet_times_figure(5.0, 1.5, [0, 3, 9]),
    "ring-seam.png": lambda: quantization_plots.ring_figure(2.5),
    "box-wall.png": lambda: quantization_plots.box_figure(2.6),
    "ring-box-oscillator.png": lambda: quantization_plots.comparison_figure(),
    "hydrogen-levels.png": lambda: quantization_plots.hydrogen_levels_figure(4),
    "belt-2pi.png": lambda: belt_plots.belt_figure(2 * math.pi),
    "belt-4pi.png": lambda: belt_plots.belt_figure(4 * math.pi),
    "ball-2pi.png": lambda: belt_plots.single_turn_ball_figure(),
    "ball-4pi.png": lambda: belt_plots.double_turn_ball_figure(math.pi / 2),
}


@click.command()
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path), default=DEFAULT_OUT)
def main(out_dir):
    """Write every chapter figure into OUT_DIR."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        fig = make()
        fig.savefig(out_dir / name, dpi=170, bbox_inches="tight")
        plt.close(fig)
        click.echo(name)


if __name__ == "__main__":
    main()
