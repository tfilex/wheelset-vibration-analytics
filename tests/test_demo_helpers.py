from src.demo.mock_data import DASHBOARD_RUL, build_checks_history


def test_build_checks_history_has_expected_columns():
    history = build_checks_history()

    assert list(history.columns) == ["Время", "Узел", "Статус", "RUL"]
    assert len(history) > 0
    assert 0 <= DASHBOARD_RUL <= 100
