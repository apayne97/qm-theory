"""Write the slide figures of the polar-coordinates section (section 2 of spherical_harmonics_and_hydrogen.py) and of the
drum Fourier example (end of the drum part of section 3) as PNG and SVG files, and the build-up of the drum sum as a
GIF: white background, 20 pt text, thick lines. The coordinate figures are 6 inches high, the drum rows 4.4.

    pixi run python standing_waves/export_figures.py [--out standing_waves/figures]
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
from drum_fourier_plots import (
    drum_fourier_error_figure,
    drum_fourier_figure,
    drum_fourier_gif_frames,
    drum_fourier_modes_figure,
    drum_fourier_steps_figure,
)
from hydrogen_plots import central_levels_figure
from coordinates_plots import (
    cartesian_cell_figure,
    length_factors_figure,
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
    "central_length_factors": length_factors_figure,
    "central_cartesian_cell": cartesian_cell_figure,
    "central_levels_l0": lambda: central_levels_figure(0, slide=True),
    "central_levels_l1": lambda: central_levels_figure(1, slide=True),
    "central_levels_l2": lambda: central_levels_figure(2, slide=True),
    "drum_fourier_steps": drum_fourier_steps_figure,
    "drum_fourier_rose_steps": lambda: drum_fourier_steps_figure("rose", (1, 10, 50, 200)),
    "drum_fourier_error": drum_fourier_error_figure,
    "drum_fourier_example": lambda: drum_fourier_figure("poke", 10, slide=True),
    "drum_fourier_modes_used": lambda: drum_fourier_modes_figure(10, slide=True, l_max=7, nr_max=6),
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
    frames = drum_fourier_gif_frames("poke")
    frames[0].save(out_dir / "drum_fourier_build_up.gif", save_all=True, append_images=frames[1:], duration=450, loop=0,
                   disposal=2)
    click.echo("drum_fourier_build_up.gif")
    frames = drum_fourier_gif_frames("rose", ms=(1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 200))
    frames[0].save(out_dir / "drum_fourier_rose_build_up.gif", save_all=True, append_images=frames[1:], duration=450,
                   loop=0, disposal=2)
    click.echo("drum_fourier_rose_build_up.gif")
    click.echo(f"{2 * len(FIGURES) + 2} files in {out_dir}")


if __name__ == "__main__":
    main()
