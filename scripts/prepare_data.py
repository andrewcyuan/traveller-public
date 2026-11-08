"""Download pinned public Small or Medium inputs and build native source-only stores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f152293e235517d504809563c833d7190b8c713b'
BASE = f'https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/{REVISION}'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(name, raw, checksums=None):
    target = raw/name
    expected = (checksums or {}).get(name)
    if checksums is not None and not expected:
        raise ValueError(f'Missing official checksum: {name}')
    if target.exists() and (not expected or digest(target) == expected):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.partial')
    subprocess.run(['curl','--location','--fail','--silent','--show-error','--retry','5','--retry-all-errors','--continue-at','-','--output',str(partial),f'{BASE}/{name}?download=true'],check=True)
    if expected:
        assert digest(partial) == expected, f'Checksum mismatch: {name}'
    partial.replace(target)
    print(json.dumps({'downloaded':name,'bytes':target.stat().st_size}),flush=True)
    return target


def extract(bundle, raw, needed):
    with tarfile.open(bundle,'r|gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or '..' in parts or len(parts)<2:
                continue
            name = 'screenshots/'+'/'.join(parts[-2:])
            if name not in needed:
                continue
            target = raw/name
            if target.exists():
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial = target.with_suffix('.partial')
            with archive.extractfile(member) as source, partial.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            partial.replace(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tier',choices=('small','medium'),default='small')
    args=parser.parse_args()
    data=args.output.resolve();data.mkdir(parents=True,exist_ok=True);raw=data.with_name(data.name+'-downloads');raw.mkdir(parents=True,exist_ok=True)
    tier_file = data/'dataset-tier.json'
    if tier_file.exists() and json.loads(tier_file.read_text()) != args.tier:
        raise ValueError('Use a separate output directory for each dataset tier')
    tier_file.write_text(json.dumps(args.tier)+'\n')
    cached=sum(p.stat().st_size for base in (data,raw) for p in base.rglob('*') if p.is_file() and not p.is_symlink())
    remaining=max(1024**3,15*1024**3-cached)
    assert shutil.disk_usage(data).free > remaining, f'Need {remaining/1024**3:.1f} GiB additional free space for dataset preparation'
    checksums={}
    for line in download('checksums.sha256',raw).read_text().splitlines():
        value,name=line.split(maxsplit=1);checksums[name.lstrip('*').removeprefix('./')]=value
    haystacks=json.loads(download(f'haystacks/lme_v2_{args.tier}.json',raw,checksums).read_text())
    questions=[json.loads(line) for line in download('questions.jsonl',raw,checksums).read_text().splitlines()]
    histories={}
    for q in questions:
        ids=haystacks[q['id']]
        assert len(ids)==len(set(ids))
        assert len(ids)==100 if args.tier=='small' else 100<len(ids)<=500
        histories.setdefault(q['domain'],set()).update(ids)
    histories={domain:sorted(ids) for domain,ids in histories.items()}
    assert len(questions) == len(haystacks) == 451
    if args.tier == 'small':
        assert all(len(ids) == 100 for ids in histories.values())
    needed_ids={i for ids in histories.values() for i in ids}
    # Stream the large JSONL; do not hold all raw histories in memory.
    selected={}
    for line in download('trajectories.jsonl',raw,checksums).open():
        trajectory=json.loads(line)
        if trajectory['id'] in needed_ids:
            selected[trajectory['id']]=trajectory
    assert set(selected)==needed_ids
    screenshots={s['screenshot'] for t in selected.values() for s in t['states']}
    bundles=['trajectory_screenshots/web_screenshots.tar.gz','trajectory_screenshots/enterprise_screenshots_base.tar.gz']
    # Also honor an official patch if present in the pinned checksum inventory.
    bundles += [n for n in checksums if n.endswith('enterprise_screenshots_patch.tar.gz')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        archives=list(pool.map(lambda name:download(name,raw,checksums),bundles[:2]))
    for archive in archives:
        extract(archive,raw,screenshots)
    for name in bundles[2:]:
        # A patch must replace base files, so extract to its own directory first.
        patch=raw/'patch';patch.mkdir(exist_ok=True)
        extract(download(name,raw,checksums),patch,screenshots)
        for path in (patch/'screenshots').glob('*/*.png'):
            target=raw/'screenshots'/path.parent.name/path.name
            if target.is_symlink():target.unlink()
            shutil.copy2(path,target)
    assert all((raw/name).is_file() for name in screenshots), 'Missing source screenshots'
    sys.path.insert(0,str(ROOT/'vendor/LongMemEval-V2/memory_modules'))
    from trajectory_store import prepare_trajectory_insert,materialize_prepared_trajectory
    for domain,ids in histories.items():
        for tid in ids:
            trajectory=selected[tid];assert trajectory['domain']==domain
            prepared=prepare_trajectory_insert(trajectory,trajectories_root_dir=raw)
            target=data/'histories'/domain/'trajectories'/tid
            materialize_prepared_trajectory(prepared,target)
        print(json.dumps({'domain':domain,'trajectories':len(ids)}),flush=True)
    for q in questions:
        if q.get('image'):
            target=data/q['image'];target.parent.mkdir(parents=True,exist_ok=True)
            source=download(q['image'],raw,checksums)
            shutil.copy2(source,target)
    shutil.copy2(raw/'questions.jsonl',data/'questions.jsonl')
    (data/'haystacks').mkdir(exist_ok=True)
    shutil.copy2(raw/f'haystacks/lme_v2_{args.tier}.json',data/f'haystacks/lme_v2_{args.tier}.json')
    if args.tier=='medium': (data/'medium-histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    report={'dataset_revision':REVISION,'tier':args.tier,'questions':len(questions),'histories':{d:len(ids) for d,ids in histories.items()},'screenshots':len(screenshots),'source_trajectory_sha256':digest(raw/'trajectories.jsonl'),'official_checksums_verified':True}
    (data/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.path.insert(0,str(ROOT))
    from delta_lme.dataset import validate_medium
    if args.tier=='medium': validate_medium(data,questions)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
