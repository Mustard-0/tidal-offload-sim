"""Freeze acceptance only after matching passing tests, archive audit and timing."""
import argparse
import json
import time
from pathlib import Path
from .runtime import ROOT, source_hashes, runtime_info, json_write, torch
from .protocol import plan, digest
from .mechanism import ARCHIVE_SHA256
from .readiness import APPROVAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--timing', type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    tests, audit, timing = [json.loads(path.read_text(encoding='utf-8')) for path in (a.tests,a.audit,a.timing)]
    hashes, runtime = source_hashes(), runtime_info('cpu')
    assert tests['component_checks_passed'] and all(c['returncode']==0 for c in tests['checks'])
    assert tests['source_hashes'] == hashes == timing['source_hashes'], 'Stale source evidence'
    assert tests['runtime'] == runtime == timing['runtime'], 'Runtime changed'
    assert audit['sha256'] == ARCHIVE_SHA256 and audit['zip_crc_passed'] and audit['extracted_files_match_archive']
    assert all(m['ast_equal'] and m['equal_except_trailing_newlines'] for m in audit['vendor_matches'].values())
    assert timing['sequential_total_hours'] > 0
    if APPROVAL.exists():
        raise FileExistsError('Preserve prior acceptance; no overwrite')
    json_write(APPROVAL, dict(passed=True, accepted_epoch=time.time(), source_hashes=hashes,
        plan_hash=digest(plan()), runtime=runtime, archive_sha256=ARCHIVE_SHA256,
        evidence=dict(tests=str(a.tests.resolve()), audit=str(a.audit.resolve()), timing=str(a.timing.resolve())),
        scope='One finite 11-cell CPU screen; no other studies; no ranking-based changes',
        active_wait='Conservative immediate-release version; not evidence for learned waiting',
        scientific_claim='Software preflight only; does not imply convergence or TMAPO superiority'))
    print(APPROVAL)


if __name__ == '__main__':
    main()
