"""The one colour palette of the qm-theory notebooks, figures and knowledge-base chapter.

Colours follow the 2026.10 QM Study Group slides (theme "Slipstream" plus a few custom colours): one colour per
quantity. Colour is used in figures and in stand-alone (display) equations only, never in running text.

Import the constants from here; do not copy hex values into a notebook or a plotting module. The LaTeX
definitions for the knowledge-base chapter are generated from the same table by `latex_definitions()`.
"""

# name -> (hex, LaTeX colour name, meaning)
_TABLE = {
    "H_COLOR": ("#4E67C8", "qmH", "height H; real part of psi"),
    "A_COLOR": ("#F14124", "qmA", "size A of a wave; marks and highlights"),
    "PHASE_COLOR": ("#7030A0", "qmPhase", "phase theta, phase difference delta"),
    "OMEGA_COLOR": ("#FF8021", "qmOmega", "angular frequency omega, and the energy E = hbar omega"),
    "FREQ_COLOR": ("#5DCEAF", "qmFreq", "frequency f"),
    "PERIOD_COLOR": ("#DF2CB2", "qmPeriod", "period T and wavelength lambda"),
    "K_COLOR": ("#6B913C", "qmK", "wave number k, and the momentum p = hbar k"),
    "CX_COLOR": ("#5ECCF3", "qmCx", "complex wave or amplitude (psi)"),
    "R_COLOR": ("#FF84FA", "qmR", "distance r from a source"),
    "IM_COLOR": ("#A7EA52", "qmIm", "imaginary part of psi"),
    "PROB_COLOR": ("#C5CAE9", "qmProb", "probability density |psi|^2"),
    "KIN_COLOR": ("#E3A008", "qmKin", "kinetic energy"),
    "POT_COLOR": ("#8C564B", "qmPot", "potential energy"),
    "INK": ("#212745", "qmInk", "dark text colour of the theme"),
}

globals().update({name: value[0] for name, value in _TABLE.items()})

LAMBDA_COLOR = PERIOD_COLOR   # period and wavelength share one colour
RE_COLOR = H_COLOR            # real part of psi is a height
ARROW_COLOR = CX_COLOR        # an arrow is a complex number

# Darker or lighter shades of one hue, to tell two curves of the same kind apart.
H2_COLOR = "#9AAAE8"
CX2_COLOR = "#2A9DC9"
OMEGA2_COLOR = "#B84F00"
K2_COLOR = "#3E5A1F"

# The belt trick: the two faces of the belt, and the plate.
FRONT_COLOR = A_COLOR
BACK_COLOR = H_COLOR
PLATE_COLOR = "#7D8A95"


def latex_definitions():
    """The `\\definecolor` lines of the knowledge-base chapter, one per entry of the table."""
    return "\n".join(
        f"\\definecolor{{{tex}}}{{HTML}}{{{hex_[1:]}}}  % {meaning}" for hex_, tex, meaning in _TABLE.values()
    )
