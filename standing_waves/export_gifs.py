"""Write every animation of spherical_harmonics_and_hydrogen.py to GIF files, for pasting into slides.

Covers every value the notebook sliders can reach: the string for n = 1 ... 8, and the drumhead for
m_l = -6 ... 6 and k = 1 ... 4 (the slider ranges of the notebook).

    pixi run python standing_waves/export_gifs.py [--out standing_waves/gifs]
"""

from pathlib import Path

import click
from hydrogen_plots import drum_gif, string_gif

N_STRING = range(1, 9)  # slider n
M_DRUM = range(-6, 7)  # slider m_l
K_DRUM = range(1, 5)  # slider k


@click.command()
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path(__file__).parent / "gifs",
              show_default=True, help="Output folder (created if missing).")
def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f"string_n{n:02d}.gif", string_gif, (n,)) for n in N_STRING]
    jobs += [(f"drum_m{m:+d}_k{k}.gif", drum_gif, (m, k)) for m in M_DRUM for k in K_DRUM]
    for name, func, args in jobs:
        (out_dir / name).write_bytes(func(*args))
        click.echo(name)
    click.echo(f"{len(jobs)} files in {out_dir}")


if __name__ == "__main__":
    main()
