import json

import click
import pytest

import sync_chapter


def test_sync_copies_then_detects_destination_edit(tmp_path, monkeypatch):
    src = tmp_path / "chapter"
    (src / "figures").mkdir(parents=True)
    (src / "quantum-mechanics.tex").write_text("a")
    (src / "quantum-mechanics.pdf").write_bytes(b"p")
    (src / "figures" / "f.png").write_bytes(b"png")
    monkeypatch.setattr(sync_chapter, "HERE", src)
    dest = tmp_path / "kb"
    record = tmp_path / "rec.json"

    names = sync_chapter.sync(dest, record)
    assert names == ["quantum-mechanics.tex", "quantum-mechanics.pdf", "figures/f.png"]
    assert (dest / "figures" / "f.png").read_bytes() == b"png"
    assert set(json.loads(record.read_text())) == set(names)

    (src / "quantum-mechanics.tex").write_text("b")
    sync_chapter.sync(dest, record)  # unchanged destination: fine
    assert (dest / "quantum-mechanics.tex").read_text() == "b"

    (dest / "quantum-mechanics.tex").write_text("edited in the knowledge base")
    with pytest.raises(click.ClickException):
        sync_chapter.sync(dest, record)
    sync_chapter.sync(dest, record, force=True)
    assert (dest / "quantum-mechanics.tex").read_text() == "b"
