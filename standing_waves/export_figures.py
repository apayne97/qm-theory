"""Write the slide figures of the polar-coordinates section (section 2 of spherical_harmonics_and_hydrogen.py) as PNG
and SVG files: white background, 20 pt text, thick lines, every figure 6 inches high.

    pixi run python standing_waves/export_figures.py [--out standing_waves/figures]
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
from coordinates_plots import (
    polar_grid_figure,
    polar_point_figure,
    spherical_coordinates_figure,
    spherical_surface_figure,
    spherical_surfaces_figure,
)

FIGURES = {
    "polar_2d_point": polar_point_figure,
    "polar_2d_grid": polar_grid_figure,
    "spherical_3d_coordinates": spherical_coordinates_figure,
    "spherical_3d_const_r": lambda: spherical_surface_figure("r"),
    "spherical_3d_const_theta": lambda: spherical_surface_figure("theta"),
    "spherical_3d_const_phi": lambda: spherical_surface_figure("phi"),
    "spherical_3d_surfaces": spherical_surfaces_figure,
}


@click.command()
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path(__file__).parent / "figures",
              show_default=True, help="Output folder (created if missing).")
@click.option("--dpi", default=200, show_default=True, help="Resolution of the PNG files.")
def main(out_dir, dpi):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        fig = make()
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, facecolor="white")
        plt.close(fig)
        click.echo(name)
    click.echo(f"{2 * len(FIGURES)} files in {out_dir}")


if __name__ == "__main__":
    main()
