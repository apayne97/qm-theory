"""Regenerate the figures of the knowledge-base chapter "Quantum chemistry: from waves in a box to DFT labels"
(chapter2/quantum-chemistry.tex) from the notebook plotting modules.

Run this after changing `qmstyle.py` or a plotting function, then rebuild the chapter.

    pixi run python make_chapter2_figures.py [OUT_DIR]
"""

import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
for folder in ("standing_waves", "dft"):
    sys.path.insert(0, str(HERE / folder))

import basis_sets_plots as bsp  # noqa: E402
import cc_plots  # noqa: E402
import dft_plots  # noqa: E402
import hydrogen_plots as hp  # noqa: E402
import molecules_plots as mp  # noqa: E402
import standing_waves_plots as swp  # noqa: E402
from cc_numerics import load_cc_json  # noqa: E402
from dft_data import load  # noqa: E402

DEFAULT_OUT = HERE / "chapter2" / "figures"

scf, ladder, disp, basis = load("scf"), load("ladder"), load("dispersion"), load("basis")
cc = load_cc_json()

FIGURES = {
    "box-1d.png": lambda: swp.line_figure(3),
    "box-2d.png": lambda: swp.box2d_figure(2, 3, 1.0),
    "box-kspace.png": lambda: swp.kspace_figure(1.0, 2, 3),
    "box-aspect-fan.png": lambda: swp.aspect_fan_figure(),
    "box-counting.png": lambda: swp.counting_figure(),
    "harmonics-gallery.png": lambda: hp.harmonic_gallery_figure(3),
    "hydrogen-central-levels.png": lambda: hp.central_levels_figure(1, slide=True),
    "hydrogen-orbital-gallery.png": lambda: hp.orbital_gallery_figure(),
    "bridge-born-oppenheimer.png": lambda: mp.born_oppenheimer_figure(0, 1.0),
    "bridge-helium.png": lambda: mp.helium_figure(1.6875),
    "bridge-screening.png": lambda: mp.screening_figure(0.5),
    "bridge-hybrids.png": lambda: mp.hybrid_figure(),
    "bridge-methane.png": lambda: mp.methane_figure(0),
    "basis-h2plus.png": lambda: bsp.h2plus_figure(2.0),
    "basis-sto-fit.png": lambda: bsp.sto_fit_figure(3),
    "dft-line-fields.png": lambda: dft_plots.line_fields_figure(scf),
    "dft-range-split.png": lambda: dft_plots.split_figure(scf["omega"], scf["short_range_exchange"], scf["long_range_exchange"]),
    "dft-bsse.png": lambda: dft_plots.bsse_figure(basis),
    "dft-water-dimer.png": lambda: dft_plots.eint_figure(ladder),
    "dft-methane-dimer.png": lambda: dft_plots.dispersion_figure(disp),
    "cc-determinants.png": lambda: cc_plots.determinant_growth_figure(cc["fci_counts"]),
    "cc-size-consistency.png": lambda: cc_plots.size_consistency_figure(cc["h2_dimer"]["scan"], cc["h2_dimer"]["monomer"]["e_ccsd_corr"]),
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
