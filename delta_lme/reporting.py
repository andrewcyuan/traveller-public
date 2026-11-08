"""Export only saved evidence and answers to the shared offline viewer."""
import gzip
import json
from pathlib import Path
from .checkpoint import fingerprint
from .memory import source_records
from .ranking import utility
from .client import READER


def read(path):
    if not path.exists(): return None
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream: return json.load(stream)


def save(path, value):
    with gzip.open(path, 'wt') as stream: json.dump(value, stream, ensure_ascii=False)


def record(root, profile, question, data, manifest):
    folder = root / profile / 'items' / question['id']
    retrieval = read(folder / 'retrieval.json') or {}
    answer = read(folder / 'answer.json') or {}
    grade = read(folder / 'grade.json') or {}
    context = read(folder / 'reader-context.json.gz')
    texts = [p['value'] for p in context or [] if p['type'] == 'text']
    ranking = read(folder / 'workspace/ranking.json')
    pool = None
    if ranking:
        sources = {r['id']: r['text'] for r in source_records(folder/'workspace/trajectories')}
        pool = [{'id': tid, 'text': sources[tid], 'rank': i, 'score': score / 3,
                 'raw_jev_utility': score} for i, (tid, score) in enumerate(ranking['ordered_jobs'], 1)]
        with gzip.open(folder / 'calls.jsonl.gz', 'rt') as stream:
            requests = [c['request'] for c in map(json.loads, stream) if c['model'] == 'typesafe/jev-1.13']
        sent = [v for req in requests for k,v in req['state'].items() if k.startswith('candidate_')]
        for item in pool:
            if ranking['decisions'][item['id']].get('reason') != 'complete_source_exceeds_payload_limit':
                assert item['text'] in sent
    expansions = []
    search = folder / 'workspace/state-search-results.jsonl.gz'
    if search.exists():
        with gzip.open(search, 'rt') as stream:
            for call_index, call in enumerate(map(json.loads, stream)):
                for item in call['candidates']:
                    src = data / 'histories' / question['domain'] / 'trajectories' / item['trajectory_id'] / 'trajectory.json'
                    state = json.loads(src.read_text())['states'][item['state_index']]
                    assert state['text'] == item['text']
                    expansions.append({**item, 'id': f"search-{call_index}-{item['id']}", 'score': None,
                                       'provenance': {'query':call['query'], 'raw_bm25_score':item['score'],
                                                      'trajectory_id':item['trajectory_id'], 'state_index':item['state_index'],
                                                      'source_sha256':item['source_sha256']}})
    errors = [read(p) for p in sorted(folder.glob('*-error.json'))]
    row = {'schema_version':1, 'run_id':profile, 'label':'Delta '+profile,
           'history':question['domain'], 'question_id':question['id'], 'question':question['question'],
           'config':{'profile':profile, 'controller':manifest['profiles'][profile], 'reader':READER,
                     'prompt_sha256':manifest['prompt_sha256'], 'query_seconds':retrieval.get('query_seconds'),
                     'saved_image_context':[p for p in context or [] if p['type'] != 'text'],
                     'reader_image_limit':manifest.get('reader_image_limit'),
                     'actual_reader_request_file':'reader-request.json.gz' if (folder/'reader-request.json.gz').exists() else None},
           'candidates':pool, 'source_expansions':expansions if search.exists() else None,
           'selected_items':[{'id':str(i),'text':text} for i,text in enumerate(texts)] if context is not None else None,
           'context':'\n\n'.join(texts), 'answer':answer.get('answer',''),
           'score':grade.get('score',0 if errors else None), 'reader_failure':json.dumps(errors) if errors else None,
           'tokens':retrieval.get('context_tokens')}
    save(folder / 'viewer-record.json.gz', row)
    return row
