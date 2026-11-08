"""Validate and scope Medium histories to the exact official per-question sets."""
import fcntl
import hashlib
import json


def validate_medium(data, questions):
    haystacks = json.loads((data/'haystacks/lme_v2_medium.json').read_text())
    histories = json.loads((data/'medium-histories.json').read_text())
    assert set(haystacks) == {q['id'] for q in questions}
    for domain in ('web', 'enterprise'):
        ids = histories[domain]
        assert len(ids) == len(set(ids))
        actual = {p.parent.name for p in (data/'histories'/domain/'trajectories').glob('*/trajectory.json')}
        assert actual == set(ids), f'Incomplete {domain} Medium source store'
        selected = [haystacks[q['id']] for q in questions if q['domain'] == domain]
        assert set().union(*map(set,selected)) == actual, f'Mismatched {domain} source union'
        for subset in selected:
            assert len(subset) == len(set(subset)) and 100 < len(subset) <= 500
            assert set(subset) <= actual, 'Cross-domain or missing trajectory'


def source_for_question(data, output, question):
    source = data/'histories'/question['domain']/'trajectories'
    if not (data/'medium-histories.json').exists():
        return source
    ids = json.loads((data/'haystacks/lme_v2_medium.json').read_text())[question['id']]
    key = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    folder = output/'haystacks'/key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'scope.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        scoped = folder/'trajectories'
        scoped.mkdir(exist_ok=True)
        for tid in ids:
            assert (source/tid/'trajectory.json').is_file(), tid
            target = scoped/tid
            if not target.exists():
                target.symlink_to((source/tid).resolve(), target_is_directory=True)
            assert target.resolve() == (source/tid).resolve()
        assert {p.parent.name for p in scoped.glob('*/trajectory.json')} == set(ids)
    return scoped
