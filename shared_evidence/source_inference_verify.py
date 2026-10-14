"""Streaming independent fit/prediction/score replay of source-inference ledgers.

Shares the versioned simulator and serialization contract, but does not call
production fitting, prediction, scoring, or experiment functions to replay them.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

from .simulate import SimulationConfig, derive_seed, simulate_question


def probability(x: float) -> float:
    z = math.exp(-abs(x))
    return 1/(1+z) if x >= 0 else z/(1+z)


def loss(x: float, y: int) -> float:
    z = x*(1-2*y)
    return max(z,0)+math.log1p(math.exp(-abs(z)))


def slope(totals: list[float], labels: list[int]) -> float:
    scale = max(map(abs,totals))
    if not scale:
        return 0.0
    def derivative(a):
        return math.fsum(t/scale*(probability(a*t) if y == 0 else -probability(-a*t))
                         for t,y in zip(totals,labels))
    if derivative(0) >= 0:
        return 0.0
    if derivative(1) <= 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low+high)/2
        if derivative(mid) < 0:
            low = mid
        else:
            high = mid
    return (low+high)/2


def independent_fit(rows: list[dict]) -> dict:
    """Only forecast logits and training outcomes are consumed here."""
    matrix = [row['log_odds'] for row in rows]
    labels = [row['outcome'] for row in rows]
    m,n = len(rows),len(matrix[0])
    residuals, norms = [],[]
    for i in range(n):
        scale = max(abs(row[i]) for row in matrix)
        col = [(2*y-1)*row[i]/scale if scale else 0 for row,y in zip(matrix,labels)]
        mean = math.fsum(x/m for x in col)
        resid = [x-mean for x in col]
        residuals.append(resid)
        norms.append(math.sqrt(math.fsum(x*x for x in resid)))
    threshold = min(1.0, math.sqrt(2*math.log(n*(n-1)/.05)/(m-1))) if n>1 else 1.0
    corr = [[0.0]*n for _ in range(n)]
    components = [{i} for i in range(n)]
    for i in range(n):
        corr[i][i] = float(norms[i] > 0)
        for j in range(i):
            r = math.fsum((a/norms[i])*(b/norms[j]) for a,b in zip(residuals[i],residuals[j])) if norms[i] and norms[j] else 0.0
            r = max(-1.0,min(1.0,r))
            corr[i][j] = corr[j][i] = r
            if r>threshold or (norms[i]>0 and all(row[i]==row[j] for row in matrix)):
                left = next(c for c in components if i in c)
                right = next(c for c in components if j in c)
                if left is not right:
                    left.update(right)
                    components.remove(right)
    components.sort(key=min)
    groups = [next(k for k,c in enumerate(components) if i in c) for i in range(n)]
    pairs = [corr[i][j] for i in range(n) for j in range(i) if groups[i]==groups[j]]
    rho = max(0.0,min(1.0,math.fsum(pairs)/len(pairs))) if pairs else 0.0
    weights = [1/(1+(groups.count(g)-1)*rho) for g in groups]
    transformed = [math.fsum(w*x for w,x in zip(weights,row)) for row in matrix]
    return {'source': {'group_ids':groups,'overlap':rho,'discount':slope(transformed,labels),
                       'correlations':corr,'threshold':threshold,'residual_df':m-1},
            'scalar': slope([math.fsum(row) for row in matrix],labels)}


def assert_close(actual, expected, path='root'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():
            raise ValueError(f'{path}: keys mismatch')
        for key in expected:
            assert_close(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,(list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):
            raise ValueError(f'{path}: length mismatch')
        for i,(a,b) in enumerate(zip(actual,expected)):
            assert_close(a,b,path+f'[{i}]')
    elif isinstance(expected,float):
        if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=2e-12,abs_tol=2e-12):
            raise ValueError(f'{path}: numerical mismatch')
    elif actual != expected:
        raise ValueError(f'{path}: value mismatch')


def predictions(logits, state, truth, rho):
    def logsumexp(values):
        maximum = max(values)
        return maximum+math.log(math.fsum(math.exp(x-maximum) for x in values))
    source = state['source']
    groups = source['group_ids']
    # Match the oracle's summation grouping, independently of its helper.
    sums = [math.fsum(logits[i] for i,g in enumerate(truth) if g==label)
            /(1+(truth.count(label)-1)*rho) for label in dict.fromkeys(truth)]
    return {'prior':0.0,
        'arithmetic_mean':logsumexp([-loss(x,1) for x in logits])-logsumexp([-loss(x,0) for x in logits]),
        'independent':math.fsum(logits), 'scalar':state['scalar']*math.fsum(logits),
        'inferred_groups':source['discount']*math.fsum(x/(1+(groups.count(g)-1)*source['overlap']) for x,g in zip(logits,groups)),
        'grouped_oracle':math.fsum(sums)}


def verify(summary_path: Path, ledger_path: Path, lock_path: Path | None = None) -> dict:
    summary = json.loads(summary_path.read_text())
    config = summary['config']
    if summary.get('status') == 'locked_synthetic_evaluation':
        if lock_path is None:
            raise ValueError('locked evaluation verification requires its lock')
        from scripts.source_inference_lock import verify_lock
        from .source_inference_experiment import ROOT
        lock = json.loads(lock_path.read_text())
        matches = [ROOT / name for name, digest in lock['config_sha256'].items()
                   if digest == summary['config_sha256']]
        if len(matches) != 1 or json.loads(matches[0].read_text()) != config:
            raise ValueError('summary config absent from lock')
        if verify_lock(lock_path, matches[0]) != summary['lock_sha256']:
            raise ValueError('summary lock mismatch')
    from .source_inference_experiment import source_hashes
    if source_hashes()!=summary['source_sha256'] or platform.python_version()!=summary['python_version']:
        raise ValueError('source or runtime mismatch')
    digest = hashlib.sha256()
    with ledger_path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    if digest.hexdigest()!=summary['ledger_receipt']['sha256']:
        raise ValueError('ledger file checksum mismatch')
    payload = hashlib.sha256()
    count = prediction_count = 0
    with gzip.open(ledger_path,'rb') as file:
        header = file.readline(); payload.update(header)
        header = json.loads(header)
        if header.get('type')!='header' or header.get('schema_version')!=1:
            raise ValueError('invalid header')
        meta = header['metadata']
        for key in ('config','source_sha256','python_version'):
            assert_close(meta[key],summary[key])
        expected_streams = {}; expected_keys=[]; used_seeds=set()
        for raw in config['conditions']:
            simulation = {k:raw[k] for k in ('n_forecasters','signal_strength','overlap','group_ids')}
            for rep in range(config['replicates']):
                key = f"{raw['name']}:{rep}"; expected_keys.append(key)
                for role in ('train','test'):
                    seed=derive_seed(config['seed'],config['phase'],raw['name'],rep,role)
                    if seed in used_seeds: raise ValueError('duplicate stream')
                    used_seeds.add(seed)
                    expected_streams[f'{key}:{role}']={'simulation':simulation,'seed':seed,'split':role,'role':role,'count':config[role+'_count'],'start':0}
        assert_close(meta['streams'],expected_streams)
        if set(meta['fits'])!=set(expected_keys) or len(summary['runs'])!=len(expected_keys):
            raise ValueError('incomplete fits or runs')
        for run, key in zip(summary['runs'],expected_keys):
            train=[]; buckets={}; signs={'scalar_opposite':0,'inferred_opposite':0,'scalar_zero':0,'inferred_zero':0,'oracle_zero':0,'scalar_opposite_repaired':0}
            name,rep=key.rsplit(':',1)
            assert_close([run['condition'],run['replicate']],[name,int(rep)])
            for role in ('train','test'):
                stream_id=key+':'+role; stream=expected_streams[stream_id]
                if role=='test':
                    state=independent_fit(train)
                    assert_close(meta['fits'][key],state,'fit')
                    assert_close(run['fit'],state,'run fit')
                for index in range(stream['count']):
                    line=file.readline(); payload.update(line); count+=1
                    row=json.loads(line)
                    q=simulate_question(SimulationConfig(**stream['simulation']),seed=stream['seed'],split=role,question_id=index)
                    expected=json.loads(json.dumps({'type':'question','stream':stream_id,**asdict(q)}))
                    if {k:row.get(k) for k in expected}!=expected:
                        raise ValueError('generated question mismatch')
                    if role=='train':
                        if set(row)!=set(expected): raise ValueError('unexpected training predictions')
                        train.append({'log_odds':row['log_odds'],'outcome':row['outcome']})
                    else:
                        logits=predictions(q.log_odds,state,list(q.group_ids),stream['simulation']['overlap'])
                        pred={method:{'logit':x,'probability':probability(x),'log_loss':loss(x,q.outcome),'brier':probability(x*(1-2*q.outcome))**2} for method,x in logits.items()}
                        expected['predictions']=pred; assert_close(row,expected,'prediction')
                        prediction_count+=len(pred)
                        for method,values in pred.items():
                            bucket=buckets.setdefault(method,{'log_loss':[],'brier':[]})
                            for metric in bucket: bucket[metric].append(values[metric])
                        s,i,o=(logits[m] for m in ('scalar','inferred_groups','grouped_oracle'))
                        signs['scalar_opposite']+=s*o<0; signs['inferred_opposite']+=i*o<0
                        signs['scalar_opposite_repaired']+=s*o<0 and i*o>0
                        for label,x in (('scalar',s),('inferred',i),('oracle',o)): signs[label+'_zero']+=x==0
            assert_close(run['scores'],{m:{k:math.fsum(v)/len(v) for k,v in b.items()} for m,b in buckets.items()},'scores')
            assert_close(run['sign_counts'],signs,'signs')
            groups=state['source']['group_ids']; truth=list(q.group_ids)
            errors=sum((groups[i]==groups[j])!=(truth[i]==truth[j]) for i in range(len(groups)) for j in range(i))
            pairs=len(groups)*(len(groups)-1)//2; rho=stream['simulation']['overlap']
            assert_close(run['diagnostics'],{'group_count':len(set(groups)),'overlap':state['source']['overlap'],
                'exact_recovery':errors==0 if rho>0 and stream['simulation']['signal_strength']>0 else None,'pair_error':errors/pairs if rho>0 and stream['simulation']['signal_strength']>0 and pairs else None},'diagnostics')
        trailer=json.loads(file.readline())
        assert_close(trailer,{'type':'trailer','count':count,'payload_sha256':payload.hexdigest()})
        if file.read(1): raise ValueError('extra ledger bytes')
    if count!=summary['ledger_receipt']['count'] or payload.hexdigest()!=summary['ledger_receipt']['payload_sha256']:
        raise ValueError('receipt mismatch')
    if source_hashes()!=summary['source_sha256']:
        raise ValueError('sources changed during verification')
    return {'status':'verified','questions_checked':count,'predictions_checked':prediction_count,
            'fits_checked':len(expected_keys),
            'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            'ledger_sha256':summary['ledger_receipt']['sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--lock',type=Path)
    args=parser.parse_args(); receipt=verify(args.summary,args.ledger,args.lock)
    text=json.dumps(receipt,indent=2)+'\n'
    if args.output: args.output.write_text(text)
    print(text,end='')


if __name__=='__main__': main()
