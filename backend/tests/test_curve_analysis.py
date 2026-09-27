import pytest

from app.curve_analysis import calculate_threshold_bands


def test_calculate_single_interpolated_band() -> None:
    curve = [
        {"frequency": 0.0, "s11": -5.0},
        {"frequency": 1.0, "s11": -15.0},
        {"frequency": 2.0, "s11": -15.0},
        {"frequency": 3.0, "s11": -5.0},
    ]
    bands = calculate_threshold_bands(curve, -10.0)
    assert len(bands) == 1
    assert bands[0]["start_frequency"] == pytest.approx(0.5)
    assert bands[0]["end_frequency"] == pytest.approx(2.5)
    assert bands[0]["bandwidth"] == pytest.approx(2.0)
    assert bands[0]["center_frequency"] == pytest.approx(1.5)


def test_calculate_multiple_bands_and_edge_band() -> None:
    curve = [
        {"frequency": 0.0, "s11": -12.0},
        {"frequency": 1.0, "s11": -8.0},
        {"frequency": 2.0, "s11": -12.0},
        {"frequency": 3.0, "s11": -8.0},
    ]
    bands = calculate_threshold_bands(curve, -10.0)
    assert len(bands) == 2
    assert bands[0]["start_frequency"] == pytest.approx(0.0)
    assert bands[0]["end_frequency"] == pytest.approx(0.5)
    assert bands[1]["start_frequency"] == pytest.approx(1.5)
    assert bands[1]["end_frequency"] == pytest.approx(2.5)


def test_empty_curve_has_no_band() -> None:
    assert calculate_threshold_bands([], -10.0) == []
