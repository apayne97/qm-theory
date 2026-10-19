"""Numbered, referenceable equations for the marimo notebooks, written as in LaTeX.

Put ``\\label{eq:name}`` inside a displayed equation and ``\\eqref{eq:name}`` (or ``\\ref{eq:name}``) in the
running text. ``numbers`` reads the notebook source once and numbers the labelled equations in order of
appearance, restarting at each ``## N Title`` heading, so the equations of section 7 are 7.1, 7.2, and so on.
``md`` then replaces each label by ``\\tag{N.k}`` (which the math renderer shows at the right of the equation)
and each reference by its number.

In a notebook:

    from eqrefs import equations_md
    md = equations_md(mo, __file__)      # use md(r\"\"\"...\"\"\") in place of mo.md(r\"\"\"...\"\"\")

Labels are never shown as such, so the same text can be copied into a LaTeX chapter.
"""

import re
from pathlib import Path

_TOKEN = re.compile(r"^[ \t]*## (\d+) |\\label\{([^}]+)\}", re.M)
_LABEL = re.compile(r"\\label\{([^}]+)\}")
_REF = re.compile(r"\\(eqref|ref)\{([^}]+)\}")


def number_labels(source):
    """Map each label in the source text to its number "N.k", counting from 1 within each ``## N`` section."""
    numbers = {}
    section, count = 0, 0
    for match in _TOKEN.finditer(source):
        if match.group(1):
            section, count = int(match.group(1)), 0
            continue
        name = match.group(2)
        if name in numbers:
            raise ValueError(f"label {name!r} is used twice")
        count += 1
        numbers[name] = f"{section}.{count}"
    return numbers


def render(text, numbers):
    """Replace labels by tags and references by numbers. A reference to an unknown label is a KeyError."""

    def label(match):
        return rf"\tag{{{numbers[match.group(1)]}}}"

    def ref(match):
        number = numbers[match.group(2)]
        return f"({number})" if match.group(1) == "eqref" else number

    return _REF.sub(ref, _LABEL.sub(label, text))


def equations_md(mo, notebook_path):
    """Return a function like ``mo.md`` that understands ``\\label`` and ``\\eqref`` for the given notebook."""
    numbers = number_labels(Path(notebook_path).read_text())

    def md(text, **kwargs):
        return mo.md(render(text, numbers), **kwargs)

    return md
