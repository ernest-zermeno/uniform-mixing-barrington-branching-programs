# Uniform Mixing in Barrington Branching Programs

Ernest Darell Zermeño — Universidad Panamericana.

Accepted in **IACR Communications in Cryptology, Volume 3, Issue 3**.

This repository contains the final-version preparation of October 3, 2026.
Editorial production is still pending; this is not a claim that the article
has already been published or that copy-editing has been completed.

The paper proves exact target-independence and quantitative uniform mixing
for terminal outputs of a fixed valid balanced transposition-template
Barrington program under fresh uniform inconsistent-path sampling.
The level-3 contraction bound is **0.0355**, supported by an exact integer
certificate. See the manuscript for the experiment, hypotheses and bounds.

## Paper and reproducibility material

| Location | Contents |
|---|---|
| [paper/main.pdf](paper/main.pdf) | Current clean manuscript |
| [paper/main.tex](paper/main.tex) | LaTeX source, with bibliography and class files in the same directory |
| [artifact/README.md](artifact/README.md) | Instructions for all 20 verification scripts |
| [artifact/CERTIFICATE_LEVEL3.md](artifact/CERTIFICATE_LEVEL3.md) | Exact level-3 certificate |
| [dist/](dist/) | Source ZIP, reproducibility ZIP, manuscript PDF and SHA-256 manifests |

The manuscript uses `iacrcc` v0.78 with `version=final`. The local volume
and issue placeholders are filled by the journal's publication system.

## Compile the manuscript

With a LaTeX installation providing `latexmk`, run:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The dependencies needed by the manuscript are included in `paper/`.

## Run the verification scripts

Use Python 3.9 or later and install the numerical dependencies in your
preferred Python environment:

```bash
python3 -m pip install -r artifact/requirements.txt
python3 artifact/scripts/20_level3_exact_certificate.py
```

The expected final line is `LEVEL-3 EXACT RATIONAL CERTIFICATE: PASS`.
The [artifact instructions](artifact/README.md) explain the other scripts,
the reduced censuses, and the distinction between numerical cross-checks
and the exact certificate.

## Regenerate the delivery archives

From the repository root:

```bash
python3 tools/build_release.py --check
python3 tools/build_release.py
```

`--check` verifies that the committed archives and PDF match the source
files; it does not modify the repository. The default command regenerates
the two ZIPs and manifests and copies the current `paper/main.pdf` into
`dist/`. Compile the manuscript first after changing its LaTeX sources.
The journal source archive has `main.tex` at its root.

## Licenses

The verification code retains its [MIT license](LICENSE). The manuscript
declares CC BY 4.0. The bundled LaTeX class and style files retain their
respective upstream license notices.
