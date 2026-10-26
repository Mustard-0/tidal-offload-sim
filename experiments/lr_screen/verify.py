"""Run bounded component checks and relevant existing regressions with logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .runtime import ROOT, json_write, source_hashes, runtime_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT/".runtime"), str(ROOT/"model"), str(ROOT)))
    temporary = a.out.resolve()/"temporary"
    temporary.mkdir()
    env['TMP'] = env['TEMP'] = str(temporary)
    hashes = source_hashes()
    evidence = []
    for name, modules in (("transfer_integration", ["experiments.lr_screen.test_mechanism"]),
                          ("components", ["experiments.lr_screen.test_components", "experiments.lr_screen.test_campaign"]),
                          ("existing_regressions", ["model.test_protocol", "model.test_review_fixes"])):
        started = time.monotonic()
        command = [sys.executable, "-m", "unittest", *modules, "-v"]
        with (a.out/(name+".log")).open("w", encoding="utf-8") as f:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
        evidence.append(dict(name=name, returncode=result.returncode, wall_s=time.monotonic()-started,
                             command=command))
    started = time.monotonic()
    command = [sys.executable, '-m', 'unittest', 'test_transfer', '-v']
    reference_env = env.copy()
    reference_env['PYTHONPATH'] = str(ROOT/'.runtime')
    with (a.out/'original_transfer.log').open('w', encoding='utf-8') as f:
        result = subprocess.run(command, cwd=ROOT/'evidence/transfer_source_20261025/mechanism_transfer',
            env=reference_env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
    evidence.append(dict(name='original_transfer', returncode=result.returncode,
                         wall_s=time.monotonic()-started, command=command))
    passed = all(r["returncode"] == 0 for r in evidence) and hashes == source_hashes()
    json_write(a.out/"verification.json", dict(component_checks_passed=passed,
        formal_training_ready=False, transfer_integrated=True,
        missing="source-locked acceptance and timing, not the transfer archive",
        source_hashes=hashes, runtime=runtime_info("cpu"), checks=evidence,
        statement="Transfer integration and component/legacy regressions only; NOT convergence or LR results"))
    print(a.out/"verification.json", "component pass", passed, "formal ready", False)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
