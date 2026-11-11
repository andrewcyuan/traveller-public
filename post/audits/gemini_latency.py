"""Offline reconstruction from saved wall clocks and monotonic call durations."""
import datetime
import gzip
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'longmemeval-v2-delta/runs'
ORIGINAL = ROOT / 'direct-pair-20261030/gemini/state-search/items'
CORRECTED = ROOT / 'failure-repairs-20261108/gemini/state-search/items'
# Explicitly audited interruption boundaries, not a general slow-query cutoff.
BOUNDARIES = {
    'aa4a4b7b': {'2026-10-30T18:50:58.243339+00:00': 'budget-stop/resume'},
    'aa64a8bd': {'2026-10-30T18:50:58.243572+00:00': 'budget-stop/resume'},
    'aca57021': {'2026-10-30T18:51:37.822210+00:00': 'budget-stop/resume'},
    '07a0145f': {
        '2026-10-30T23:17:16.502198+00:00': 'shared wall/monotonic clock gap',
        '2026-10-30T23:38:07.094525+00:00': 'shared wall/monotonic clock gap',
    },
    '07ab3723': {
        '2026-10-30T23:17:16.498771+00:00': 'shared wall/monotonic clock gap',
        '2026-10-30T23:52:58.919532+00:00': 'shared wall/monotonic clock gap',
    },
}

def calls(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]

def record(folder):
    corrected = CORRECTED / folder.name
    timing_path = corrected / 'retrieval.json'
    if not timing_path.exists():
        timing_path = folder / 'retrieval.json'
    timing = json.loads(timing_path.read_text())['query_seconds']
    corrected_calls = corrected / 'calls.jsonl.gz'
    newer = calls(corrected_calls) if corrected_calls.exists() else []
    call_path = corrected_calls if any(r.get('model') == 'gemini-3.8-flash' for r in newer) else folder / 'calls.jsonl.gz'
    rows = newer if call_path == corrected_calls else calls(call_path)
    retrieval_calls = [r for r in rows if r.get('model') in ('gemini-3.8-flash', 'typesafe/jev-1.13')]
    gaps = []
    for previous, current in zip(retrieval_calls, retrieval_calls[1:]):
        reason = BOUNDARIES.get(folder.name, {}).get(current['at'])
        if reason:
            elapsed = (datetime.datetime.fromisoformat(current['at']) - datetime.datetime.fromisoformat(previous['at'])).total_seconds()
            gaps.append({'at': current['at'], 'reason': reason, 'excluded_seconds': elapsed - current.get('seconds', 0)})
    assert len(gaps) == len(BOUNDARIES.get(folder.name, {}))
    adjusted = timing - sum(g['excluded_seconds'] for g in gaps)
    assert adjusted > 0
    return {'id': folder.name, 'wall_seconds': timing, 'active_estimate_seconds': adjusted,
            'excluded_intervals': gaps, 'timing_path': str(timing_path), 'calls_path': str(call_path),
            'timing_sha256': hashlib.sha256(timing_path.read_bytes()).hexdigest(),
            'calls_sha256': hashlib.sha256(call_path.read_bytes()).hexdigest()}

if __name__ == '__main__':
    rows = [record(folder) for folder in sorted(ORIGINAL.iterdir()) if folder.is_dir()]
    assert len(rows) == 451 and len({r['id'] for r in rows}) == 451
    result = {'questions': len(rows), 'wall_mean_seconds': statistics.mean(r['wall_seconds'] for r in rows),
              'active_mean_estimate_seconds': statistics.mean(r['active_estimate_seconds'] for r in rows),
              'excluded_interval_count': sum(len(r['excluded_intervals']) for r in rows),
              'limitation': 'Reconstructed estimate, not a fresh monotonic query measurement. Boundary subtraction also removes unmeasured local overhead between the last pre-interruption call and the next call. All measured call durations and other query time remain included.',
              'records': rows}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
