"""Read-only archive/source audit; writes provenance, never runs experiments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from .protocol import ROOT
import sys
sys.path.insert(0, str(ROOT/'.runtime'))
from .mechanism import ARCHIVE_SHA256


def sha(path, normalized=False):
    if normalized:
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    transfer = ROOT/'evidence/transfer_source_20261025/mechanism_transfer'
    digest = sha(a.archive)
    assert digest == ARCHIVE_SHA256, 'Unexpected archive bytes'
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None, 'Archive CRC failure'
        count = len(z.infolist())
        for item in z.infolist():
            if item.is_dir():
                continue
            dest = (transfer.parent/item.filename).resolve()
            assert transfer.parent.resolve() in dest.parents, 'Unsafe archive path'
            with z.open(item) as f:
                assert hashlib.sha256(f.read()).hexdigest() == sha(dest), item.filename
    pairs = [('support/location_methods.py', 'vendor/location_methods.py'),
             ('legacy/accountant.py', 'vendor/accountant.py')]
    matches = {}
    for src,dst in pairs:
        left, right = transfer/src, ROOT/'experiments/lr_screen'/dst
        matches[src] = dict(normalized_byte_equal=sha(left,True)==sha(right,True),
            equal_except_trailing_newlines=left.read_text().rstrip('\n')==right.read_text().rstrip('\n'),
            ast_equal=ast.dump(ast.parse(left.read_text()))==ast.dump(ast.parse(right.read_text())),
            source_sha256=sha(left,True), vendor_sha256=sha(right,True))
    assert all(m['equal_except_trailing_newlines'] and m['ast_equal'] for m in matches.values())
    source = {str(p.relative_to(transfer)): sha(p, True)
              for p in transfer.rglob('*.py') if '__pycache__' not in str(p)}
    original = json.loads((transfer/'config.json').read_text())['environment']
    from .protocol import configuration
    current, _ = configuration('TMAPO', 1)
    differences = {k:dict(transfer=original.get(k), rl=current.get(k))
                   for k in sorted(set(original)|set(current)) if original.get(k) != current.get(k)}
    physics = {name: dict(transfer=sha(transfer/'support/base'/name, True),
                         rl=sha(ROOT/'model'/name, True),
                         equal_except_trailing_newlines=(transfer/'support/base'/name).read_text().rstrip('\n') ==
                                                        (ROOT/'model'/name).read_text().rstrip('\n'))
               for name in ('environment.py','physics.py','privacy.py','workload.py')}
    out = dict(archive=str(a.archive), sha256=digest, zip_crc_passed=True,
        extracted_files_match_archive=True, entries=count, vendor_matches=matches,
        source_hashes=source, config_differences=differences, base_source_comparison=physics,
        selection=json.loads((transfer/'selection.json').read_text()),
        integrated='Privacy only; RL actor and true physics retained; no greedy policy imported',
        limitations=['Conditional single-batch accountant, not trajectory/transcript DP',
                     'Active WAIT disabled pending causal physical-benefit estimator',
                     'Archive greedy selection at epsilon1 is Center; user-fixed s0.5 is not reselected',
                     'Archive runtime differs; compatibility checked in current RL runtime, no old results relabeled'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(a.out)


if __name__ == '__main__':
    main()
