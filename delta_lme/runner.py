"""Fresh and resumable native evaluation with explicit paths and budget."""
import fcntl
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from .budget import BudgetExceeded, BudgetLedger
from .checkpoint import write_json, fingerprint, canonical_json
from .client import Client, READER
from .grading import grade as native_grade
from .memory import PROFILE, query_evidence, decode, ensure_summaries
from .native import harness
from .state_search import build_index
from .reader import native_text, first_native
from .reader_images import image_limit, limit_images
from .redaction import strip_images
from .reporting import read, save, record

REPO = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def now(): return datetime.now(UTC).isoformat()


def prepare(data, output, profiles, budget, workers, tier='small'):
    questions = [json.loads(line) for line in (data / 'questions.jsonl').read_text().splitlines()]
    assert len(questions) == len({q['id'] for q in questions}) == 451
    if tier == 'medium':
        from .dataset import validate_medium
        validate_medium(data, questions)
    if tier == 'small':
        for domain in ('web','enterprise'):
            ensure_summaries(data / 'histories' / domain / 'trajectories')
    sources = sorted(p for p in data.rglob('*') if p.is_file() and p.suffix != '.sqlite')
    code = sorted({*REPO.glob('delta_lme/**/*.py'), *REPO.glob('strategies/delta/*.py'),
                   *REPO.glob('strategies/delta/scripts/*.py'), *REPO.glob('vendor/LongMemEval-V2/**/*.py'),
                   *REPO.glob('evals/run_viewer/*.py'), *REPO.glob('evals/run_viewer/*.html'),
                   REPO/'pyproject.toml', REPO/'uv.lock', REPO/'.python-version', REPO/'scripts/prepare_data.py'})
    identity = {'data_root':str(data), 'ids':[q['id'] for q in questions],
                'inputs':{q['id']:fingerprint(q) for q in questions},
                'source_hashes':{str(p.relative_to(data)):sha(p) for p in sources},
                'code_hashes':{str(p.relative_to(REPO)):sha(p) for p in code},
                'prompt_sha256':{p.name:sha(p) for p in (REPO/'strategies/delta').glob('*.txt')},
                'instructions_sha256':sha(REPO/'strategies/delta/INSTRUCTION.md'),
                'profiles':{p:{'model':os.environ.get('DELTA_CONTROLLER_MODEL','openai/gpt-6-luna'),
                               'provider':os.environ.get('DELTA_CONTROLLER_PROVIDER','openrouter'),
                               'effort':'high',
                               'max_turns':48,'max_output_tokens':8192} for p in profiles},
                'reader':READER, 'reader_image_limit':image_limit(READER['model']), 'budget_usd':budget, 'workers_per_profile':workers}
    if tier != 'small':
        identity.update(tier=tier, haystack_policy='Exact per-question official trajectory subset; private FTS index construction included in query timing')
    path = output / 'manifest.json'
    if path.exists(): assert read(path)['identity'] == identity, 'Frozen run configuration or inputs changed'
    else: write_json(path, {'created_at':now(),'identity':identity})
    for domain in ('web','enterprise'):
        source = data / 'histories' / domain / 'trajectories'
        assert len(list(source.glob('*/trajectory.json'))) == (100 if tier == 'small' else len(read(data/'medium-histories.json')[domain]))
        if tier == 'small': ensure_summaries(source)
        if tier == 'small' and 'state-search' in profiles:
            build_index(source, output / f'{domain}-states.sqlite')
    return questions,identity


def item(root, profile, q):
    folder=root/profile/'items'/q['id'];folder.mkdir(parents=True,exist_ok=True);return folder


def retrieve(root, profile, q, data, budget):
    folder=item(root,profile,q)
    if any((folder/name).exists() for name in ('retrieval.json','retrieval-error.json')): return
    start=folder/'query-start.json'
    if not start.exists():write_json(start,{'wall_start':time.time()})
    try:
        from .dataset import source_for_question
        source=source_for_question(data,root,q)
        value=query_evidence(Client(root,folder,profile,budget),
                             source,folder/'workspace',q['question'],
                             data/q['image'] if q.get('image') else None,
                             state_index=None if (data/'medium-histories.json').exists() else root/f"{q['domain']}-states.sqlite")
        save(folder/'reader-context.json.gz',value['context'])
        write_json(folder/'retrieval.json',{'id':q['id'],**{k:v for k,v in value.items() if k!='context'},
                                          'context_sha256':fingerprint(value['context']),
                                          'query_seconds':time.time()-read(start)['wall_start']})
    except BudgetExceeded: raise
    except Exception as error:
        write_json(folder/'retrieval-error.json',{'id':q['id'],'error':repr(error),'score':0,
                                                 'query_seconds':time.time()-read(start)['wall_start']})
    finally:
        # Medium indexes are query-local scratch; saved source evidence and search logs remain.
        if (data/'medium-histories.json').exists() and any((folder/n).exists() for n in ('retrieval.json','retrieval-error.json')):
            (folder/'workspace/state-index.sqlite').unlink(missing_ok=True)


def messages(q, context, data):
    packet = harness.build_messages(harness.get_system_prompt(q['domain']),q['question'],
                                  str(data/q['image']) if q.get('image') else None,context)[0]
    return limit_images(packet, image_limit(READER['model']))[0]


def answer(root,profile,q,data,budget):
    folder=item(root,profile,q)
    if not (folder/'retrieval.json').exists() or any((folder/n).exists() for n in ('answer.json','answer-error.json')):return
    try:
        packet=messages(q,read(folder/'reader-context.json.gz'),data)
        save(folder/'reader-request.json.gz',packet)
        response=Client(root,folder,profile,budget).post('v1/chat/completions',{**READER,'messages':packet})
        text=native_text(response)
        assert isinstance(text,str) and text.strip(), 'First native reader text is empty'
        write_json(folder/'answer.json',{'id':q['id'],'answer':text,'messages_sha256':fingerprint(packet),'usage':response.get('usage')})
    except BudgetExceeded:raise
    except Exception as error:write_json(folder/'answer-error.json',{'id':q['id'],'score':0,'error':repr(error)})


def grade(root,profile,q,budget):
    folder=item(root,profile,q)
    if not (folder/'answer.json').exists() or any((folder/n).exists() for n in ('grade.json','grade-error.json')):return
    try:write_json(folder/'grade.json',native_grade(q,read(folder/'answer.json')['answer'],Client(root,folder,profile,budget)))
    except BudgetExceeded:raise
    except Exception as error:write_json(folder/'grade-error.json',{'id':q['id'],'score':0,'error':repr(error)})


def batch(root,profile,phase,questions,operation,workers,data,manifest):
    write_json(root/profile/'progress.json',{'phase':phase,'finished':0,'total':len(questions),'at':now()})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(operation,q):q for q in questions}
        try:
            for count,future in enumerate(as_completed(futures),1):
                future.result();record(root,profile,futures[future],data,manifest)
                value={'profile':profile,'phase':phase,'finished':count,'total':len(questions),'at':now()}
                write_json(root/profile/'progress.json',value);print(json.dumps(value),flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise


def verify_case(root,profile,q,data):
    folder=root/profile/'items'/q['id'];calls=[]
    if (folder/'calls.jsonl.gz').exists():
        with gzip.open(folder/'calls.jsonl.gz','rt') as stream:calls=list(map(json.loads,stream))
    first,stats=first_native(calls,READER['model'])
    assert stats.get('unused_reader_attempts',0)==0
    retrieval=read(folder/'retrieval.json');saved=read(folder/'answer.json')
    if retrieval:
        current=decode(folder/'workspace',q['id'])
        assert fingerprint(current['context'])==retrieval['context_sha256']==fingerprint(read(folder/'reader-context.json.gz'))
    if saved:
        assert first and native_text(first['response'])==saved['answer']
        packet=messages(q,current['context'],data)
        assert fingerprint(packet)==saved['messages_sha256']
        assert first['request']==strip_images({**READER,'messages':packet})
    value=read(folder/'grade.json')
    if value:
        class Replay:
            def post(self,route,payload):
                replies=[c['response'] for c in calls if not c.get('cache_hit') and 'error' not in c
                         and c['route']==route and c['request']==strip_images(payload)
                         and c.get('response',{}).get('choices')]
                assert len(replies)==1
                return replies[0]
        assert native_grade(q,saved['answer'],Replay())==value
    else:assert list(folder.glob('*-error.json')),q['id']
    timing=retrieval or read(folder/'retrieval-error.json')
    return {'id':q['id'],'domain':q['domain'],'category':q['question_type'],
            'score':value['score'] if value else 0,'query_seconds':timing['query_seconds']}


def run_profile(root,profile,questions,data,budget,workers,manifest):
    folder=root/profile;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            def finish_question(q):
                retrieve(root,profile,q,data,budget)
                answer(root,profile,q,data,budget)
                grade(root,profile,q,budget)
            saved=[q for q in questions if (item(root,profile,q)/'retrieval.json').exists()]
            batch(root,profile,'saved-answer-grade',saved,finish_question,workers,data,manifest)
            batch(root,profile,'retrieve-answer-grade',questions,finish_question,workers,data,manifest)
            batch(root,profile,'reader',questions,lambda q:answer(root,profile,q,data,budget),workers,data,manifest)
            frozen={str(p.relative_to(folder)):sha(p) for p in folder.glob('items/*/answer.json')}
            path=folder/'frozen-answers.json'
            if path.exists():assert read(path)==frozen
            else:write_json(path,frozen)
            batch(root,profile,'judge',questions,lambda q:grade(root,profile,q,budget),workers,data,manifest)
            rows=[verify_case(root,profile,q,data) for q in questions]
            records=folder/'viewer-runs.jsonl.gz'
            with gzip.open(records,'wt') as stream:
                for q in questions:stream.write(json.dumps(record(root,profile,q,data,manifest))+'\n')
            subprocess.run([sys.executable,'-m','evals.run_viewer',str(records),'--output',str(folder/'run-viewer.html'),
                            '--expected-runs',str(len(questions))],check=True,cwd=REPO)
            summary={'complete':True,'profile':profile,'questions':len(rows),'correct':sum(r['score'] for r in rows),
                     'mean_query_seconds':sum(r['query_seconds'] for r in rows)/len(rows),'results':rows,
                     'first_native_replies_and_native_grades_verified':True}
            write_json(folder/'audit.json',summary);write_json(folder/'job.json',{'status':'complete','at':now()})
        except BaseException as error:
            write_json(folder/'job.json',{'status':'stopped','error':repr(error),'at':now()});raise


def run(data,output,budget,workers=2,prepare_only=False,tier='small'):
    profiles = (PROFILE,)
    data,output=Path(data).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    assert budget>0 and workers>0
    with (output/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        questions,manifest=prepare(data,output,profiles,budget,workers,tier)
        if prepare_only:
            print(json.dumps({'prepared':True,'questions':len(questions),'profiles':profiles,'paid_calls':0}));return
        ledger=BudgetLedger(output/'budget.json',budget)
        assert not ledger.snapshot()['reservations'],'Reconcile interrupted calls before resuming'
        run_profile(output,PROFILE,questions,data,budget,workers,manifest)
        # Recheck immutable code and inputs after execution.
        prepare(data,output,profiles,budget,workers,tier)
        assert not ledger.snapshot()['reservations']
        combined=output/'viewer-runs.jsonl.gz'
        with gzip.open(combined,'wt') as out:
            for profile in profiles:
                with gzip.open(output/profile/'viewer-runs.jsonl.gz','rt') as src:
                    for line in src:out.write(line)
        subprocess.run([sys.executable,'-m','evals.run_viewer',str(combined),'--output',str(output/'run-viewer.html'),
                        '--expected-runs',str(451*len(profiles))],check=True,cwd=REPO)
        summary={'complete':True,'budget':ledger.snapshot(),'profiles':{p:read(output/p/'audit.json') for p in profiles}}
        write_json(output/'summary.json',summary)
        (output/'research-journal.md').write_text('# Delta evaluation batch\n\n'+now()+'\n\n'+
             '\n'.join(f"- {p}: {a['correct']}/451, {a['mean_query_seconds']:.3f}s mean query." for p,a in summary['profiles'].items())+
             '\n\n[Offline viewer](run-viewer.html)\n')
