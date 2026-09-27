#!/usr/bin/env python3
"""Package the manuscript and verification artifact, or check committed packages."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ZIP_DATE = (2026, 9, 26, 0, 0, 0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(mapping):
    stream = io.BytesIO()
    hashes = {}
    with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as output:
        for name, source in sorted(mapping.items()):
            data = source.read_bytes()
            info = ZipInfo(name, ZIP_DATE)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)
            hashes[name] = digest(data)
    return stream.getvalue(), hashes


def delivery_files():
    paper = ROOT / 'paper'
    artifact = ROOT / 'artifact'
    sources = {name: paper / name for name in
               ('main.tex', 'refs.bib', 'iacrcc.cls', 'after-hyperref.sty')}
    scripts = sorted((artifact / 'scripts').glob('*.py'))
    if [p.name.split('_', 1)[0] for p in scripts] != [f'{i:02}' for i in range(1, 21)]:
        raise SystemExit('Expected exactly the 20 numbered verification scripts.')
    scientific = {f'scripts/{p.name}': p for p in scripts}
    scientific.update({name: artifact / name for name in
                       ('README.md', 'CERTIFICATE_LEVEL3.md',
                        'THEOREM_dimensional_obstruction.md', 'requirements.txt', 'LICENSE')})
    files, contents = {}, {}
    for name, mapping in [('sources.zip', sources), ('artifact.zip', scientific)]:
        files[name], contents[name] = archive(mapping)
    files['paper.pdf'] = (paper / 'main.pdf').read_bytes()
    files['ARCHIVE_CONTENTS.json'] = (json.dumps(contents, indent=2) + '\n').encode()
    files['SHA256SUMS.txt'] = ''.join(
        f'{digest(data)}  {name}\n' for name, data in sorted(files.items())
    ).encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='verify dist/ without writing files')
    args = parser.parse_args()
    expected = delivery_files()
    output = ROOT / 'dist'
    if args.check:
        errors = []
        for name, data in expected.items():
            target = output / name
            if not target.is_file():
                errors.append(f'missing: {name}')
            elif target.read_bytes() != data:
                errors.append(f'out of date: {name}')
        extras = set(p.name for p in output.iterdir()) - set(expected) if output.exists() else set()
        errors.extend(f'unexpected delivery file: {name}' for name in sorted(extras))
        if errors:
            raise SystemExit('\n'.join(errors))
        print('RELEASE ARCHIVES AND MANUSCRIPT: PASS')
    else:
        output.mkdir(exist_ok=True)
        for name, data in expected.items():
            (output / name).write_bytes(data)
        print('Built source ZIP, 25-file artifact ZIP, PDF and SHA-256 manifests.')


if __name__ == '__main__':
    main()
