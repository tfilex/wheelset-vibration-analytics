import numpy as np

from src.utils.health_index import (
    compute_hi,
    compute_slope,
    format_rul_display,
    get_status,
    rul_to_km,
)


def test_hi_range():
    rul = np.linspace(1.0, 0.0, 100)
    hi = compute_hi(rul)
    assert np.all(hi >= 0.0)
    assert np.all(hi <= 1.0)


def test_status_levels():
    assert get_status(0.90)[0] == 1
    assert get_status(0.72)[0] == 2
    assert get_status(0.50)[0] == 3
    assert get_status(0.20)[0] == 4


def test_rul_to_km_no_negative():
    rul_km, sigma = rul_to_km(0.0, slope=-0.5)
    assert rul_km >= 0.0
    assert sigma >= 0.0


def test_rul_to_km_range():
    rul_km, sigma = rul_to_km(0.5)
    assert 400_000 < rul_km < 600_000
    assert sigma >= 0.0


def test_format_display():
    display = format_rul_display(42_000, 5_000)
    assert "42 000" in display
    assert "5 000" in display


def test_slope_negative_on_degradation():
    hi_series = np.linspace(0.9, 0.5, 50)
    assert compute_slope(hi_series, n=20) < 0.0

