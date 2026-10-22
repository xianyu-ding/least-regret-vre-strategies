"""Small deterministic checks on the offline runner's public contract."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('figure_runner', ROOT / 'scripts/reproduce_figures.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

def test_script_lists_are_disjoint_and_present():
    assert not set(runner.DIRECT).intersection(runner.RANKED)
    assert len(runner.DIRECT) == 3
    assert len(runner.RANKED) == 6
    assert all((ROOT / 'code/Figure' / name).is_file() for name in runner.DIRECT + runner.RANKED)

def test_nonpositive_draws_rejected(tmp_path):
    import pytest
    with pytest.raises(SystemExit) as exc:
        runner.main(['--n-mc', '0', '--output-dir', str(tmp_path)])
    assert exc.value.code == 2

def test_partial_blocked_exit_two_and_manifest(tmp_path, monkeypatch):
    import json
    import types
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='', stderr=''))
    out = tmp_path/'out'
    code = runner.main(['--score-dir', str(tmp_path/'absent'), '--output-dir', str(out), '--require-all'])
    assert code == 2
    result = json.loads((out/'figure_run_manifest.json').read_text())
    assert result['summary']['blocked'] == 12
    assert result['summary']['failed'] == 0

def test_failed_child_is_nonzero(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=7, stdout='', stderr='test error'))
    assert runner.main(['--score-dir', str(tmp_path/'absent'), '--output-dir', str(tmp_path/'out')]) == 1

def test_zero_exit_plot_without_outputs_is_failure(tmp_path, monkeypatch):
    import types
    score_dir=tmp_path/'scores';score_dir.mkdir()
    for source in ['govern','news','research','zhihu']:
        (score_dir/f'{source}_scores_only.jsonl').write_text('{}\n')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='', stderr=''))
    assert runner.main(['--score-dir', str(score_dir), '--output-dir', str(tmp_path/'out')]) == 1
