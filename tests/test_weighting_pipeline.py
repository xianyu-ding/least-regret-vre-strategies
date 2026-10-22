"""Offline regression tests. Synthetic fixtures are not research reproductions."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIGURES = Path(__file__).resolve().parents[1] / 'code' / 'Figure'
sys.path.insert(0, str(FIGURES))
import pipeline_common as common
import Data_weight as weights
import Data_rank as ranking


def test_score_parser_rejects_invalid_and_missing():
    for value in (None, True, 0, 6, 2.5, 'error', 'score 3', np.nan, np.inf):
        with pytest.raises(ValueError):
            common.parse_score_value(value)
    assert common.parse_score_value('3') == 3
    with pytest.raises(ValueError, match='missing='):
        common.normalize_scores_dict({})
    valid = {key: 3 for key in common.INDICATORS}
    with pytest.raises(ValueError, match='unknown='):
        common.normalize_scores_dict(dict(valid, extra=3))


def test_jsonl_parser_reports_file_line(tmp_path):
    p = tmp_path / 'scores.jsonl'
    p.write_text(json.dumps({'scores': {key: 3 for key in common.INDICATORS}}) + '\n{}\n')
    with pytest.raises(ValueError, match='scores.jsonl:2:'):
        common.read_jsonl_scores(p, 'govern')


def test_grouped_schema_requires_all_ten():
    with pytest.raises(ValueError, match='schema mismatch'):
        common.flatten_assessment({'economic': {'total_transition_cost': 3}})


def props_equal_concern():
    df = pd.DataFrame([dict(source='govern', **{key: 3 for key in common.INDICATORS})])
    return weights.per_source_tier_props(weights.collapse_to_tiers(df))


def test_plot_matches_hierarchical_weights():
    module = importlib.import_module('Fig_weighting_details_abc')
    props = props_equal_concern()
    Q = weights.aggregate_across_sources_with_lambda(props, .5)
    top = weights.calculate_top_level_weights(Q)
    plotted = module.calculate_top_level_by_source(module.calculate_source_specific_weights(props))['govern']
    assert plotted == pytest.approx(top)
    assert sum(plotted.values()) == pytest.approx(1)
    assert all(value == pytest.approx(.25) for value in plotted.values())
    overall = module.calculate_source_specific_weights(props)['govern']
    assert overall['land_occupation'] == pytest.approx(.25 / 7)
    assert sum(overall.values()) == pytest.approx(1)


def write_weights(tmp_path, monkeypatch):
    Q = weights.aggregate_across_sources_with_lambda(props_equal_concern(), .5)
    overall = weights.calculate_overall_weights(weights.calculate_top_level_weights(Q), weights.calculate_env_sub_weights(Q))
    path = tmp_path / 'govern_weights.csv'
    pd.DataFrame(overall.items(), columns=['Indicator', 'Overall_Weight']).to_csv(path, index=False)
    monkeypatch.setattr(ranking, 'WEIGHTS_DIR', tmp_path)
    return path


def test_weight_csv_round_trip(tmp_path, monkeypatch):
    write_weights(tmp_path, monkeypatch)
    top, env = ranking.load_perspective_weights('government')
    assert top.sum() == pytest.approx(1)
    assert env.sum() == pytest.approx(1)


def test_invalid_weight_alias_rejected(tmp_path, monkeypatch):
    path = write_weights(tmp_path, monkeypatch)
    df = pd.read_csv(path)
    df.loc[0, 'Indicator'] = 'unknown'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='exactly once'):
        ranking.load_perspective_weights('government')


def test_exact_ties_are_permutation_invariant():
    scores = np.array([.2, .2, .4])
    assert common.rank_scores(scores).tolist() == [1.5, 1.5, 3]
    assert common.winner_shares(scores).tolist() == [.5, .5, 0]
    permutation = [2, 0, 1]
    assert np.array_equal(common.rank_scores(scores[permutation]), common.rank_scores(scores)[permutation])
    assert np.array_equal(common.winner_shares(scores[permutation]), common.winner_shares(scores)[permutation])


def test_dirichlet_zero_support():
    draw = common.draw_dirichlet(np.random.default_rng(0), [1, 0, 0, 0], 10)
    assert np.array_equal(draw, [1, 0, 0, 0])
    with pytest.raises(ValueError):
        common.draw_dirichlet(np.random.default_rng(0), [0, 0], 10)


def test_empty_perspectives_fail_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(ranking, 'PERSPECTIVES', {})
    with pytest.raises(ValueError, match='No perspectives'):
        ranking.multi_perspective_analysis(None, tmp_path)


def test_small_synthetic_ranking_arsi_and_no_scenario_save(tmp_path, monkeypatch):
    # A is helped by Cost; removing Cost must worsen A's rank -> positive ARSI.
    df = pd.DataFrame({'alternative': ['A', 'B'], **{key: [.9, .1] for key in common.INDICATORS}})
    df['total_transition_cost'] = [.1, .9]
    top = pd.Series({'Cost': .9, 'LOLE': .03, 'Environment': .04, 'CO2': .03})
    env = pd.Series(1 / 7, index=ranking.ENV_SET)
    for name, value in [('N_MC', 4), ('KAPPA_TOP', 1e9), ('KAPPA_ENV', 1e9), ('SAVE_WEIGHT_SCENARIOS', False)]:
        monkeypatch.setattr(ranking, name, value)
    evolution, scenarios, sci, arsi = ranking.scc_with_scca_arsi_analysis(df, top, env, 'test', tmp_path)
    row = arsi[(arsi.alternative == 'A') & (arsi.criterion == 'Cost')].iloc[0]
    assert row.arsi_value == pytest.approx(1)
    assert row.interpretation == 'Strong Advantage Criterion'
    assert scenarios is None
    assert np.isfinite(sci.structural_criticality_index).all()
    wins = pd.read_csv(tmp_path / 'scc_winrates_test.csv')
    assert wins.win_rate.sum() == pytest.approx(1)
    assert (tmp_path / 'arsi_rank_shifts_sample_test.csv').exists()
