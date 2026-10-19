"""Tests for eqrefs.py, and a check that the labels and references in every notebook are consistent."""

import re
from pathlib import Path

import pytest

from eqrefs import number_labels, render

HERE = Path(__file__).resolve().parent


def test_numbering_restarts_at_each_section():
    source = "## 1 A\n$$ a \\label{eq:a} $$\n$$ b \\label{eq:b} $$\n## 2 B\n$$ c \\label{eq:c} $$\n"
    assert number_labels(source) == {"eq:a": "1.1", "eq:b": "1.2", "eq:c": "2.1"}


def test_subsection_headings_do_not_restart_the_count():
    source = "## 6 W\n$$ a \\label{eq:a} $$\n### 6.1 X\n$$ b \\label{eq:b} $$\n"
    assert number_labels(source) == {"eq:a": "6.1", "eq:b": "6.2"}


def test_render_replaces_labels_and_references():
    numbers = {"eq:a": "3.2"}
    assert render("x = 1 \\label{eq:a}", numbers) == "x = 1 \\tag{3.2}"
    assert render("see \\eqref{eq:a}", numbers) == "see (3.2)"
    assert render("equation \\ref{eq:a}", numbers) == "equation 3.2"


def test_duplicate_label_is_an_error():
    with pytest.raises(ValueError):
        number_labels("## 1 A\n\\label{eq:a}\n\\label{eq:a}\n")


def test_unknown_reference_is_an_error():
    with pytest.raises(KeyError):
        render("see \\eqref{eq:missing}", {})


def _notebooks_using_labels():
    for path in HERE.glob("*/*.py"):
        text = path.read_text()
        if "\\label{" in text or "\\eqref{" in text:
            yield path


@pytest.mark.parametrize("path", list(_notebooks_using_labels()), ids=lambda p: p.name)
def test_notebook_labels_and_references_are_consistent(path):
    text = path.read_text()
    numbers = number_labels(text)  # raises on a repeated label
    for _, name in re.findall(r"\\(eqref|ref)\{([^}]+)\}", text):
        assert name in numbers, f"{path.name}: reference to unknown label {name}"
    outside_displays = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    assert "\\label{" not in outside_displays, f"{path.name}: a label outside a displayed equation"
