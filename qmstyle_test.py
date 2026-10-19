import re

import qmstyle


def test_every_colour_is_a_hex_string():
    for name in dir(qmstyle):
        if name.endswith("_COLOR") or name == "INK":
            assert re.fullmatch(r"#[0-9A-Fa-f]{6}", getattr(qmstyle, name)), name


def test_quantities_have_distinct_colours():
    names = ["H", "A", "PHASE", "OMEGA", "FREQ", "PERIOD", "K", "CX", "R", "IM", "PROB", "KIN", "POT"]
    values = [getattr(qmstyle, f"{n}_COLOR") for n in names]
    assert len(set(values)) == len(values)


def test_latex_definitions_cover_the_table():
    text = qmstyle.latex_definitions()
    assert "\\definecolor{qmOmega}{HTML}{FF8021}" in text
    assert text.count("\\definecolor") == 14


def test_notebooks_use_only_palette_colours_in_equations():
    from pathlib import Path

    palette = {value.upper() for name, value in vars(qmstyle).items() if isinstance(value, str) and value.startswith("#")}
    for path in Path(__file__).parent.glob("*/*.py"):
        for colour in re.findall(r"\\textcolor\{(#[0-9A-Fa-f]{6})\}", path.read_text()):
            assert colour.upper() in palette, (path.name, colour)


def test_equations_are_coloured_but_running_text_is_not():
    from pathlib import Path

    for path in Path(__file__).parent.glob("*/*.py"):
        text = path.read_text()
        outside_display = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        assert "\\textcolor" not in outside_display, path.name
