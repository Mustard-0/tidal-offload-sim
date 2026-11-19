"""Bind the unchanged experiment worker to the corrected transport manifest."""
import argparse
import scalability_protocol_v2 as spec
import scalability_worker as worker


if __name__ == '__main__':
    worker.spec = spec
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gate', 'train', 'evaluate'))
    parser.add_argument('--cell', choices=spec.CELLS)
    parser.add_argument('--trace', type=int, choices=spec.TRACES)
    args = parser.parse_args()
    raise SystemExit(0 if worker.run(args.action, args.cell, args.trace) else 2)
