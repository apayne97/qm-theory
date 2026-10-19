"""Copy the quantum-mechanics chapter into a checkout of the Chodera Lab knowledge-base repo.

The chapter is edited here. The knowledge base is only a destination: this command copies the `.tex`, the PDF and
the figures into `biophysics/quantum-mechanics/` and records what it wrote in `.last_sync.json`. It refuses to run if
the destination was changed since the last sync (so an edit made there is never overwritten silently) or has
uncommitted changes. `lecture.cls` and `commons.sty` belong to the lab repo and are never copied.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = Path.home() / "science/knowledge-base-qm"
TARGET_DIR = "biophysics/quantum-mechanics"
RECORD = HERE / ".last_sync.json"


def files_to_sync():
    """Chapter files (relative to `chapter/`) that are copied to the knowledge base."""
    names = ["quantum-mechanics.tex", "quantum-mechanics.pdf"]
    names += sorted(f"figures/{p.name}" for p in (HERE / "figures").glob("*.png"))
    return names


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_since_last_sync(destination, record):
    """Destination files whose content differs from what the last sync wrote."""
    return [name for name, h in record.items() if (destination / name).exists() and digest(destination / name) != h]


def uncommitted(destination):
    """Lines of `git status --porcelain` for the destination folder, or [] if it is not in a git checkout."""
    if not destination.exists():
        return []
    out = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=destination, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if ".build" not in line]


def sync(destination, record_path=RECORD, force=False):
    """Copy the chapter files into `destination` and return the list of names copied."""
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    if not force:
        stale = changed_since_last_sync(destination, record)
        if stale:
            raise click.ClickException("changed in the knowledge base since the last sync: " + ", ".join(stale))
        dirty = uncommitted(destination)
        if dirty:
            raise click.ClickException("uncommitted changes in the knowledge base:\n" + "\n".join(dirty))
    names = files_to_sync()
    (destination / "figures").mkdir(parents=True, exist_ok=True)
    new_record = {}
    for name in names:
        (destination / name).write_bytes((HERE / name).read_bytes())
        new_record[name] = digest(destination / name)
    record_path.write_text(json.dumps(new_record, indent=1, sort_keys=True) + "\n")
    return names


@click.command()
@click.argument("checkout", type=click.Path(file_okay=False, exists=True, path_type=Path), default=DEFAULT_TARGET)
@click.option("--force", is_flag=True, help="Overwrite even if the destination changed or is dirty.")
def main(checkout, force):
    """Copy the chapter into CHECKOUT (a knowledge-base checkout), without committing or pushing."""
    names = sync(checkout / TARGET_DIR, force=force)
    click.echo(f"copied {len(names)} files to {checkout / TARGET_DIR}; review, commit and push there yourself")


if __name__ == "__main__":
    main()
