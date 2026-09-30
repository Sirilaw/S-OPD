import sys
from pathlib import Path

import pandas as pd
import pytest

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.dataset.image_mcq import WeMath  # noqa: E402
from vlmeval.dataset.image_vqa import MathVerse  # noqa: E402
from vlmeval.dataset.utils.wemath import evaluate_update_main_results_df  # noqa: E402
from vlmeval.smp.status_report import flatten_summary_metrics  # noqa: E402


def test_wemath_reports_numeric_core_strict_as_primary_metric():
    total_counts = {
        'InadequateGeneralization': 10,
        'RoteMemorization_loose': 15,
        'RoteMemorization_strict': 20,
        'InsufficientKnowledge': 25,
        'CompleteMastery_strict': 470,
        'CompleteMastery_loose': 475,
    }
    rates = {
        'InsufficientKnowledge_rate': '4.76%',
        'InadequateGeneralization_rate': '1.90%',
        'CompleteMastery_strict_rate': '95.92%',
        'RoteMemorization_strict_rate': '4.08%',
        'CompleteMastery_loose_rate': '96.94%',
        'RoteMemorization_loose_rate': '3.06%',
    }

    score = evaluate_update_main_results_df(pd.DataFrame(), total_counts, rates)
    metrics = flatten_summary_metrics(score[['Core (Strict)']])

    assert score.loc[0, 'Score (Strict)'] == '90.48%'
    assert metrics['Core (Strict)'] == pytest.approx(90.48)
    assert WeMath.report_primary_metric(metrics) == {'Core (Strict)': pytest.approx(90.48)}


def test_mathverse_reports_mean_of_five_splits():
    metrics = {
        'split=Vision Only|Overall': 10.0,
        'split=Vision Intensive|Overall': 20.0,
        'split=Vision Dominant|Overall': 30.0,
        'split=Text Lite|Overall': 40.0,
        'split=Text Dominant|Overall': 50.0,
    }

    assert MathVerse.report_primary_metric(metrics) == {'5-Split Mean': 30.0}


def test_mathverse_single_split_keeps_single_split_primary_metric():
    metrics = {'split=Vision Only|Overall': 42.0}

    assert MathVerse.report_primary_metric(metrics) == {
        'split=Vision Only|Overall': 42.0,
    }
