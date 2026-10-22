"""Shared offline paths and validated ten-criterion score input.

Environment overrides allow the original standalone scripts to share one layout.
No model calls are made. Missing data is never substituted with a concern score.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = Path(os.environ.get('LEAST_REGRET_SCORE_DIR', ROOT / 'data' / 'deepseek rating score'))
RESULTS_DIR = Path(os.environ.get('LEAST_REGRET_RESULTS_DIR', ROOT / 'results'))
WEIGHTS_DIR = RESULTS_DIR / 'weights'
SCC_DIR = RESULTS_DIR / 'scc'
FIG_DIR = RESULTS_DIR / 'figures'
INDICATOR_PATH = Path(os.environ.get('LEAST_REGRET_INDICATORS', ROOT / 'data' / 'processed' / 'combined_data.csv'))
INDICATORS = ['total_transition_cost', 'lole', 'land_occupation', 'freshwater_ecotoxicity',
              'human_toxicity', 'ionising_radiation', 'freshwater_eutrophication',
              'marine_eutrophication', 'mineral_resource_depletion', 'co2_emission']
SOURCES = ('govern', 'news', 'research', 'zhihu')


def parse_score_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f'Invalid concern score: {value!r}; require integer 1..5')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid concern score: {value!r}') from exc
    if not np.isfinite(number) or not number.is_integer() or not 1 <= number <= 5:
        raise ValueError(f'Invalid concern score: {value!r}; require integer 1..5')
    return int(number)


def normalize_scores_dict(scores):
    if not isinstance(scores, dict):
        raise ValueError('scores must be an object')
    missing = sorted(set(INDICATORS) - set(scores))
    unknown = sorted(set(scores) - set(INDICATORS))
    if missing or unknown:
        raise ValueError(f'Ten-criterion schema mismatch: missing={missing}; unknown={unknown}')
    return {key: parse_score_value(scores[key]) for key in INDICATORS}


def flatten_assessment(assessment):
    if not isinstance(assessment, dict):
        raise ValueError('assessment must be an object')
    allowed = {'economic', 'technical', 'environmental', 'global_environmental'}
    unknown = set(assessment) - allowed
    if unknown:
        raise ValueError(f'Unknown assessment categories: {sorted(unknown)}')
    flat = {}
    for category, metrics in assessment.items():
        if not isinstance(metrics, dict):
            raise ValueError(f'Assessment category {category} must contain an object')
        if set(flat).intersection(metrics):
            raise ValueError('Duplicate assessment indicators')
        flat.update(metrics)
    return normalize_scores_dict(flat)


def read_jsonl_scores(path, source_tag, include_length=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing score archive: {path}')
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('record must be an object')
                if 'scores' in obj:
                    scores = normalize_scores_dict(obj['scores'])
                elif 'assessment' in obj:
                    scores = flatten_assessment(obj['assessment'])
                else:
                    raise ValueError('record has neither scores nor assessment')
                row = {'source': source_tag, **scores}
                if include_length:
                    text = obj.get('input_text') or obj.get('question') or obj.get('answer') or ''
                    row['length'] = len(str(text))
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path.name}:{line_no}: {exc}') from exc
    if not rows:
        raise ValueError(f'Empty score archive: {path}')
    return pd.DataFrame(rows)


def rank_scores(values):
    """Average ranks for exact ties; 1 is best. No arbitrary tolerance is imposed."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Ranking requires nonempty finite scores')
    return pd.Series(values).rank(method='average', ascending=True).to_numpy()


def winner_shares(values):
    values = np.asarray(values, dtype=float)
    rank_scores(values)  # validate
    tied = values == values.min()
    return tied.astype(float) / tied.sum()


def draw_dirichlet(rng, centre, concentration):
    """Sample only positive support; zero weights stay zero across NumPy versions."""
    centre = np.asarray(centre, dtype=float)
    if not np.isfinite(centre).all() or (centre < 0).any() or centre.sum() <= 0:
        raise ValueError('Dirichlet centre must be finite, nonnegative, and nonzero')
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError('Concentration must be positive and finite')
    centre = centre / centre.sum()
    positive = centre > 0
    output = np.zeros_like(centre)
    output[positive] = rng.dirichlet(concentration * centre[positive])
    return output
