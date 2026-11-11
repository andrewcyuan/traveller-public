"""Subtract recorded unsuccessful request attempts and configured retry backoff."""
import gzip,json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3]/'longmemeval-v2-delta/runs'
def read_calls(path):
 with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]
def retry_time(rows):
 groups=defaultdict(list)
 for r in rows:
  if r.get('cache_hit'):continue
  groups[r.get('cache_key',r.get('key'))].append(r)
 removed=[]
 for group in groups.values():
  for i,r in enumerate(group):
   response=r.get('response') or {}
   bad='error' in r or (i<len(group)-1 and response.get('status') in ['incomplete','failed'])
   if not bad:continue
   # Client sleeps only before another error retry; attempt 3 is terminal.
   backoff=(30 if '429' in str(r.get('error')) else 2) if 'error' in r and r.get('attempt',3)<3 else 0
   removed.append({'at':r['at'],'model':r['model'],'attempt':r.get('attempt'),'seconds':r.get('seconds',0),'backoff_seconds':backoff})
 return removed

def main():
 prior=json.loads(Path(__file__).with_name('gemini_latency.json').read_text())
 reports={}
 for name in ['gemini','opus']:
  records=[]
  if name=='gemini':
   inputs=[(r['id'],r['active_estimate_seconds'],Path(r['calls_path'])) for r in prior['records']]
  else:
   original=ROOT/'opus-medium-20261107/state-search/items'
   inputs=[]
   for d in sorted(original.iterdir()):
    if not d.is_dir():continue
    p=d/'retrieval.json';inputs.append((d.name,json.loads(p.read_text())['query_seconds'],d/'calls.jsonl.gz'))
  for qid,seconds,path in inputs:
   rows=[r for r in read_calls(path) if r.get('model') in ['gemini-3.8-flash','typesafe/jev-1.13','anthropic/claude-opus-5.5']]
   exclusions=retry_time(rows);removed=sum(r['seconds']+r['backoff_seconds'] for r in exclusions)
   assert seconds-removed>0,(qid,seconds,removed)
   records.append({'id':qid,'previous_seconds':seconds,'excluded_attempts':exclusions,'adjusted_seconds':seconds-removed})
  assert len(records)==451
  reports[name]={'questions':451,'mean_seconds':sum(r['adjusted_seconds'] for r in records)/451,'excluded_retry_seconds':sum(r['previous_seconds']-r['adjusted_seconds'] for r in records),'records':records}
 reports['method']='Remove unsuccessful API attempt durations and configured backoff; retain final successful attempt. Model-slot queue waits are not separately logged. Retrieval has six local slots per model, with two workers in original runs and at most four in correction batch, so no slot-capacity queue wait is expected; zero queue time is an inference from concurrency limits, not a measured decomposition. Gemini downtime correction is inherited from gemini_latency.json. Reader/grader calls are excluded.'
 Path(__file__).with_suffix('.json').write_text(json.dumps(reports,indent=2)+'\n')
 print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in reports.items() if isinstance(v,dict)},indent=2))
if __name__=='__main__':main()
