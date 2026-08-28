"""Numerical agreement with the ordinary time-series benchmark implementation."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from agentad.evaluation import evaluate, get_metrics, volume_under_surface
from agentad.evaluation.affiliation import pr_from_events

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORK_ROOT = PROJECT_ROOT / "forks" / "TSB-AD"

pytestmark = pytest.mark.skipif(
    not FORK_ROOT.is_dir(), reason="TSB-AD reference fork is unavailable"
)


def _reference_metricor():
    sys.path.insert(0, str(FORK_ROOT))
    try:
        from TSB_AD.evaluation.basic_metrics import basic_metricor

        return basic_metricor()
    finally:
        sys.path.pop(0)


def _reference_get_metrics():
    sys.path.insert(0, str(FORK_ROOT))
    try:
        from TSB_AD.evaluation.metrics import get_metrics as reference

        return reference
    finally:
        sys.path.pop(0)


def _reference_affiliation():
    sys.path.insert(0, str(FORK_ROOT))
    try:
        from TSB_AD.evaluation.affiliation.metrics import pr_from_events as reference

        return reference
    finally:
        sys.path.pop(0)


def _case(length: int = 137):
    labels = np.zeros(length, dtype=int)
    labels[5:10] = 1
    labels[40:48] = 1
    labels[-12:-4] = 1
    scores = np.random.default_rng(20260828).normal(size=length)
    return labels, scores


def test_vus_matches_tsb_ad():
    labels, scores = _case()
    reference = _reference_metricor().RangeAUC_volume_opt(
        labels, scores, windowSize=9, thre=37
    )
    result = volume_under_surface(labels, scores, window_size=9, threshold_count=37)

    np.testing.assert_allclose(result.tpr, reference[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.fpr, reference[1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.precision, reference[2], rtol=0, atol=1e-14)
    assert result.roc == pytest.approx(reference[-2], abs=1e-14)
    assert result.pr == pytest.approx(reference[-1], abs=1e-14)


def test_fixed_prediction_metrics_match_tsb_ad():
    labels, scores = _case(80)
    prediction = np.asarray(scores > 0.1, dtype=int)
    reference = _reference_metricor()
    expected = {
        "Standard-F1": reference.metric_PointF1(
            labels, scores, preds=prediction.copy()
        ),
        "PA-F1": reference.metric_PointF1PA(labels, scores, preds=prediction.copy()),
        "Event-based-F1": reference.metric_EventF1PA(
            labels, scores, preds=prediction.copy()
        ),
        "R-based-F1": reference.metric_RF1(labels, scores, preds=prediction.copy()),
        "Affiliation-F": reference.metric_Affiliation(
            labels, scores, preds=prediction.copy()
        ),
    }

    result = evaluate(
        labels,
        scores,
        y_pred=prediction,
        metrics=tuple(expected),  # type: ignore[arg-type]
    )
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, abs=1e-14)


def test_oracle_suite_matches_tsb_ad():
    labels, scores = _case(80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = _reference_get_metrics()(scores, labels, slidingWindow=5, thre=17)
    result = get_metrics(scores, labels, slidingWindow=5, thre=17)

    assert tuple(result) == tuple(expected)
    for name, value in expected.items():
        assert result[name] == pytest.approx(value, abs=1e-14)


def test_affiliation_details_match_tsb_ad():
    predicted = [(1, 3), (5, 8), (12, 14)]
    ground_truth = [(2, 6), (10, 13)]
    expected = _reference_affiliation()(predicted, ground_truth, (0, 16))
    result = pr_from_events(predicted, ground_truth, (0, 16))

    assert tuple(result) == tuple(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(result[name], value, rtol=0, atol=1e-14)
