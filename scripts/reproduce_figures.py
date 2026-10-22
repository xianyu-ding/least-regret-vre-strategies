#!/usr/bin/env python3
"""Reproduce available archived-data figures without network or model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ['Fig_score_ac.py', 'Fig_weighting_details_abc.py', 'Fig_weighting_sensitivity.py']
RANKED = ['Fig_ari.py', 'Fig_scca_sci.py', 'Fig_direclet_weight.py',
          'Fig_raking_scc_winrate.py', 'Fig_ranking_scc_winrate_sensiti.py', 'Fig_没用但是好看的图.py']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-dir', type=Path, default=ROOT / 'data/deepseek rating score')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--indicators', type=Path, default=ROOT / 'data/processed/combined_data.csv')
    parser.add_argument('--n-mc', type=int, default=20000, help='Ranking draws; smaller values are smoke tests, not final estimates')
    parser.add_argument('--require-all', action='store_true', help='Exit 2 if any figure remains blocked')
    args = parser.parse_args(argv)
    if args.n_mc <= 0:
        parser.error('--n-mc must be positive')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'; logs.mkdir(exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(out / '.mplconfig'),
               XDG_CACHE_HOME=str(out / '.cache'), LEAST_REGRET_SCORE_DIR=str(args.score_dir.resolve()),
               LEAST_REGRET_RESULTS_DIR=str(out), LEAST_REGRET_INDICATORS=str(args.indicators.resolve()))
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Existing archived scores and summary data only; no upstream simulation or LLM calls',
              'n_mc': args.n_mc, 'seed': 42, 'inputs': [], 'scripts': []}
    def record(name, status, reason=None, outputs=None):
        entry = {'script': name, 'status': status}
        if reason: entry['reason'] = reason
        if outputs is not None: entry['outputs'] = outputs
        report['scripts'].append(entry)
        print(f'{status.upper()}: {name}' + (f': {reason}' if reason else ''), flush=True)
    def run(path, extra=(), figures=False):
        before = {str(p): p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.pdf','.svg'}}
        proc = subprocess.run([sys.executable, str(ROOT / path), *map(str, extra)], cwd=ROOT, env=env, capture_output=True, text=True)
        (logs / (Path(path).stem + '.log')).write_text(proc.stdout + '\n' + proc.stderr, encoding='utf-8')
        outputs = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.pdf','.svg'} and before.get(str(p)) != p.stat().st_mtime_ns]
        ok = proc.returncode == 0 and (not figures or bool(outputs))
        record(path, 'passed' if ok else 'failed', None if ok else f'Exit {proc.returncode}; see logs/{Path(path).stem}.log. No output is not treated as success.', outputs)
        return ok
    missing = []
    for source in ['govern','news','research','zhihu']:
        p = args.score_dir / f'{source}_scores_only.jsonl'
        if not p.is_file(): missing.append(str(p))
        else:
            report['inputs'].append({'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    # Record the actual supplied scenario inputs as well as the rating archives.
    extra_inputs = sorted((ROOT / 'data/raw').glob('*/*.csv'))
    if args.indicators.is_file(): extra_inputs.append(args.indicators)
    for p in extra_inputs:
        report['inputs'].append({'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    weights_ok = False
    if missing:
        for name in ['Data_weight.py', *DIRECT]: record('code/Figure/' + name, 'blocked', 'Missing archive(s): ' + ', '.join(missing))
    else:
        weights_ok = run('code/Figure/Data_weight.py')
        for name in DIRECT:
            run('code/Figure/' + name, figures=True)
    # The nine-metric table is informative, not a replacement for the ten-metric ranking input.
    run('src/02_build_mcdm_dataset.py', ['--output', out / 'energy/combined_data.csv'])
    ranking_reason = 'A real complete ten-criterion indicator table including CO2_emission is required; no CO2 values are fabricated.'
    if not args.indicators.is_file():
        for name in ['Data_rank.py', *RANKED]: record('code/Figure/' + name, 'blocked', ranking_reason)
    elif not weights_ok:
        for name in ['Data_rank.py', *RANKED]: record('code/Figure/' + name, 'blocked', 'Validated weights were not generated.')
    else:
        # Data_rank validates schema and finite numeric values; malformed supplied data is a failure.
        ranking_ok = run('code/Figure/Data_rank.py', ['--n-mc', args.n_mc])
        for name in RANKED:
            if ranking_ok: run('code/Figure/' + name, figures=True)
            else: record('code/Figure/' + name, 'blocked', 'Ranking producer failed; see its log.')
    record('code/Figure/Fig_raking_scc_knockout.py', 'blocked', 'Legacy knockout tables have no compatible producer in this repository; not interchangeable with ARSI/SCCA.')
    report['summary'] = {status: sum(x['status'] == status for x in report['scripts']) for status in ['passed','blocked','failed']}
    (out / 'figure_run_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Manifest:', out / 'figure_run_manifest.json')
    if report['summary']['failed']: return 1
    if args.require_all and report['summary']['blocked']: return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
