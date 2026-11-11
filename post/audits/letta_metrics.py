"""Reconstruct Letta metrics offline; never invoke models or change saved runs."""
import gzip
import hashlib
import json
import statistics
from pathlib import Path

ORIGINAL = Path('/tmp/lme-letta-runtime/lme-small')
PATCHED = Path('/tmp/lme-letta-runtime/lme-small-error-repair')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proxy_cost(root):
    paths = sorted(root.glob('*/receipt.json'))
    receipts = [read(p) for p in paths]
    unsettled = [p.parent for p in sorted(root.glob('*/request.json.gz'))
                 if not (p.parent / 'receipt.json').exists()]
    assert all(r['cost_usd'] is not None and r['status'] == 200 for r in receipts)
    manifest = {str(p.relative_to(root)): digest(p) for p in paths}
    return {
        'source': str(root), 'receipts': len(paths),
        'estimated_usd': sum(r['cost_usd'] for r in receipts),
        'receipt_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        'unsettled': [{'request': str(p / 'request.json.gz'),
                       'request_sha256': digest(p / 'request.json.gz'),
                       'reserve_usd': read(p / 'settings.json')['reserve_usd']} for p in unsettled],
    }


def main():
    with gzip.open(PATCHED / 'reports/patched/viewer-runs.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == len({r['question_id'] for r in rows}) == 451
    assert sum(r['score'] for r in rows) == 115
    records = []
    for row in rows:
        folder = PATCHED / 'items' / row['question_id']
        timing = folder / 'retrieval.json'
        assert read(folder / 'grade.json')['score'] == row['score']
        assert read(timing)['seconds'] == row['config']['query_seconds']
        records.append({'id': row['question_id'], 'seconds': read(timing)['seconds'],
                        'source': str(timing.resolve()), 'sha256': digest(timing)})
    stages = {
        'memory_web': proxy_cost(ORIGINAL / 'dream/web/calls'),
        'memory_enterprise': proxy_cost(ORIGINAL / 'dream/enterprise/calls'),
        'retrieval_original': proxy_cost(ORIGINAL / 'query-calls'),
        'retrieval_repairs': proxy_cost(PATCHED / 'query-calls'),
    }
    ledgers = [dict(source=str(p), sha256=digest(p), **read(p))
               for p in [ORIGINAL / 'budget.json', PATCHED / 'budget.json']]
    memory_cost = sum(stages[k]['estimated_usd'] for k in ('memory_web', 'memory_enterprise'))
    retrieval_cost = sum(stages[k]['estimated_usd'] for k in ('retrieval_original', 'retrieval_repairs'))
    reader_grader_cost = sum(r['spent_usd'] for r in ledgers)
    result = {
        'questions': 451, 'correct': 115,
        'mean_retrieval_seconds': statistics.mean(r['seconds'] for r in records),
        'memory_build_seconds': {domain: read(ORIGINAL / 'dream' / domain / 'run-exit.json')['seconds']
                                 for domain in ('web', 'enterprise')},
        'memory_build_estimated_usd': memory_cost,
        'retrieval_estimated_usd_including_repairs': retrieval_cost,
        'reader_grader_reported_usd_including_repairs': reader_grader_cost,
        'total_estimated_usd': memory_cost + retrieval_cost + reader_grader_cost,
        'unresolved_allowance_usd': sum(r['uncertain_usd'] for r in ledgers)
            + sum(r['reserve_usd'] for s in stages.values() for r in s['unsettled']),
        'method': 'Latency is the mean saved retrieval process duration for the 451 scored items, using repaired retrievals where applicable. Excludes reader, grading, memory construction, prior failed retrievals, and downtime before repairs. Cost includes both memory builds and all original/repair query, reader and grader charges, including unsuccessful attempts; excludes synthetic smoke/preflight runs. Opus costs are estimates from saved token usage at the audit proxy rates, not invoice reconciliation. Reader/grader costs are saved ledger charges. Unresolved allowances are budget reservations, not confirmed charges or guaranteed billing bounds. The web build has one request without a saved response or receipt; reader/grader ledgers retain $2.48 uncertain.',
        'opus_proxy_rates_usd_per_million': {'input': 4, 'output': 20, 'cache_read': 0.2, 'cache_creation': 8},
        'proxy_stages': stages, 'reader_grader_ledgers': ledgers, 'records': records,
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('records', 'proxy_stages', 'reader_grader_ledgers')}, indent=2))


if __name__ == '__main__':
    main()
