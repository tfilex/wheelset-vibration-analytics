from src.utils.thresholds import alert_level, is_vibration_alert, speed_threshold


def test_thresholds_by_speed():
    assert abs(speed_threshold(30) - 2.1) < 0.01
    assert abs(speed_threshold(60) - 3.0) < 0.01
    assert abs(speed_threshold(100) - 3.9) < 0.01


def test_alert_true():
    assert is_vibration_alert(4.0, 60) is True


def test_alert_false():
    assert is_vibration_alert(2.0, 60) is False


def test_alert_level():
    assert alert_level(2.0, 60) == "normal"
    assert alert_level(3.5, 60) == "warning"
    assert alert_level(6.0, 60) == "critical"

