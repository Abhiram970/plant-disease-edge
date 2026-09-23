"""
Build the Zenodo deposit and check it against what the manuscript promises.

The data-availability statement is a contract: it names the registries, the prompts and raw
replies, every result file, the per-architecture results of the supervised baselines, the
weight-space interpolation sweep, the generators and the experimental code, and it claims the
release "reproduces every number and figure in this article from the released result files".

This script treats that sentence as a test. Each promise below is a named group of paths; if a
group is empty, the build ABORTS and names it, because shipping a deposit that is missing
something the paper promises is worse than shipping late. (The statement briefly promised
training logs, which had been archived out of the repository; this check is what makes that
class of mistake impossible to repeat.)

    python scripts/prepare_zenodo_archive.py            # build dist/zenodo/
    python scripts/prepare_zenodo_archive.py --check    # verify the promises, build nothing

Output:
    dist/zenodo/plant-disease-edge-<version>.zip
    dist/zenodo/MANIFEST.sha256          one line per file, sha256 + size + path
    dist/zenodo/README.md                what the archive is and how to reproduce from it
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "dist" / "zenodo"

# Each entry is (promise as the manuscript words it, [paths]). A promise with no files is fatal.
PROMISES: list[tuple[str, list[str]]] = [
    ("the descriptor registries for every strategy",
     ["descriptors", "descriptors_cupl", "descriptors_dclip"]),
    ("the DCLIP+ and CuPL+ prompts and raw replies",
     ["manual_baselines"]),
    ("every result file",
     ["results", "docs/paper/paper_numbers.json", "docs/paper/revision_numbers.json",
      "docs/paper/revision_results"]),
    ("the per-architecture results of the 14 supervised baselines",
     ["results/supervised"]),
    ("the weight-space interpolation sweep",
     ["results/wiseft"]),
    ("the table and figure generators",
     ["docs/paper/make_tables.py", "docs/paper/make_tex_tables.py", "docs/paper/make_figures.py",
      "docs/paper/make_tex_tables_revision.py", "docs/paper/make_figures_revision.py",
      "docs/paper/paper_numbers.py"]),
    ("the full experimental code",
     ["scripts"]),
    ("the manuscript this release accompanies",
     ["docs/paper/manuscript/revision/main.tex", "docs/paper/manuscript/revision/main.pdf",
      "docs/paper/manuscript/revision/refs.bib"]),
    ("the licences and citation metadata",
     ["LICENSE", "LICENSE-DATA.md", "CITATION.cff"]),
]

# Never ship these, whatever directory they live in.
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "onnx"}
EXCLUDE_SUFFIX = {".pyc", ".pt", ".pth", ".onnx", ".aux", ".fls", ".fdb_latexmk",
                  ".synctex.gz", ".blg", ".out", ".abs", ".bbl"}

README_HEADER = """# {title}

**Version {version}** · deposited {date} · DOI: {doi}

{authors}

{keywords}

Licences: code MIT (`LICENSE`); data CC BY 4.0 (`LICENSE-DATA.md`); the manuscript is covered by
neither, see below. Cite the article and this deposit — `CITATION.cff` gives both.

## What is in this archive

{contents}

"""

README = """# Plant-disease edge VLM — data and code release

This archive accompanies *Compact Vision--Language Models for Cross-Crop Plant-Disease Diagnosis
at the Edge: A CPU-Only Study*.

It contains the descriptor registries for every authoring strategy, the DCLIP+ and CuPL+ prompts
with their raw replies, every result file the manuscript reads, the per-architecture results of
the 14 supervised baselines, the weight-space interpolation sweep, the table and figure
generators, the full experimental code, and the manuscript itself.

It does **not** contain the images. Those are the SAGE crop-disease dataset at revision
`bc9bd2899f`, released separately under the MIT licence; `scripts/config.py` pins that revision
and `scripts/sage_data.py` fetches it.

## Reproducing the numbers

Every table and figure is generated from the result files in this archive; no number in the
manuscript is transcribed by hand.

```
python docs/paper/paper_numbers.py                  # result JSONs  -> paper_numbers.json
python docs/paper/make_tex_tables_revision.py       # -> manuscript/revision/tab_*.tex
python docs/paper/make_figures_revision.py          # -> manuscript/revision/figures/*.png
cd docs/paper/manuscript/revision && latexmk -pdf main.tex
```

Re-measuring from the images, rather than re-deriving from the result files, needs a GPU and the
SAGE subset; `scripts/rescore_descriptors.py` documents the protocol and refuses to run against
anything but the 288-pixel build the published numbers were measured on.

## Licences

The code is MIT (`LICENSE`). The descriptor registries, the prompts and raw replies, and the
result files are CC BY 4.0 (`LICENSE-DATA.md`). Rights in the manuscript itself are governed by
the agreement with the publisher and by neither licence; `LICENSE-DATA.md` sets out the
boundary. Cite the article and this deposit: `CITATION.cff` gives both.

## Integrity

`MANIFEST.sha256` lists every file in this archive with its SHA-256 digest and size.
"""


def iter_files(rel: str):
    p = REPO / rel
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.parts) & EXCLUDE_DIRS) \
                    and f.suffix not in EXCLUDE_SUFFIX and not f.name.endswith(".synctex.gz"):
                yield f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the promises, build nothing")
    ap.add_argument("--version", default=date.today().isoformat())
    args = ap.parse_args()

    selected: dict[Path, None] = {}
    broken = []
    print("Checking the data-availability statement against the repository:\n")
    for promise, paths in PROMISES:
        files = [f for rel in paths for f in iter_files(rel)]
        for f in files:
            selected[f] = None
        status = "ok " if files else "MISSING"
        print(f"  [{status}] {promise}  ({len(files)} files)")
        if not files:
            broken.append((promise, paths))
    if broken:
        print("\nABORT: the manuscript promises artefacts this repository does not contain:")
        for promise, paths in broken:
            print(f"  - {promise}: none of {paths} exist")
        print("Either restore them or narrow the data-availability statement before depositing.")
        return 1

    doi_placeholder = "PENDING-ZENODO-DOI"
    tex = (REPO / "docs/paper/manuscript/revision/main.tex").read_text(encoding="utf-8")
    pending = doi_placeholder in tex
    print(f"\n  [{'note' if pending else 'ok '}] data-availability DOI: "
          + ("still PENDING-ZENODO-DOI -- mint it, then run scripts/paper_fixes/set_doi.py"
             if pending else "resolved"))

    total = sum(f.stat().st_size for f in selected)
    print(f"\n  {len(selected)} files, {total / 1e6:.1f} MB")
    if args.check:
        print("\n--check only, nothing written")
        return 0

    DIST.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(selected):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{h}  {f.stat().st_size:>10}  {f.relative_to(REPO).as_posix()}")
    (DIST / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # The deposit's front matter is built from .zenodo.json, so the README a reader opens and the
    # metadata Zenodo shows cannot drift apart.
    meta = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    authors = "\n".join(
        f"- **{c['name']}**"
        + (f" ([{c['orcid']}](https://orcid.org/{c['orcid']}))" if c.get("orcid") else "")
        + (f" — {c['affiliation']}" if c.get("affiliation") else "")
        for c in meta["creators"])
    contents = "\n".join(
        f"| {promise} | {len([f for rel in paths for f in iter_files(rel)])} |"
        for promise, paths in PROMISES)
    header = README_HEADER.format(
        title=meta["title"], version=args.version, date=date.today().isoformat(),
        doi=("not yet minted — see the article's data-availability statement"
             if pending else "see CITATION.cff"),
        authors=authors,
        keywords="Keywords: " + ", ".join(meta.get("keywords", [])),
        contents="| what the article promises | files |\n|---|---|\n" + contents)
    (DIST / "README.md").write_text(header + README.split("\n", 1)[1].lstrip("\n"),
                                    encoding="utf-8", newline="\n")

    zip_path = DIST / f"plant-disease-edge-{args.version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(selected):
            z.write(f, f.relative_to(REPO).as_posix())
        z.write(DIST / "MANIFEST.sha256", "MANIFEST.sha256")
        z.write(DIST / "README.md", "README.md")
    print(f"\nwrote {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {DIST / 'MANIFEST.sha256'}  ({len(lines)} entries)")

    if not (REPO / "LICENSE").exists():
        print("\nBEFORE PUBLISHING: this repository has no LICENSE file and Zenodo will ask for "
              "one.\n  Choose a licence, add it, and set the same value in .zenodo.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
