"""Speed-dependent vibration thresholds."""

from __future__ import annotations


def speed_threshold(speed_kmh: float, base_g: float = 3.0) -> float:
    """Return vibration RMS threshold in g for the current speed."""
    speed = float(speed_kmh)
    if speed < 40.0:
        factor = 0.70
    elif speed <= 80.0:
        factor = 1.00
    else:
        factor = 1.30
    return float(base_g) * factor


def is_vibration_alert(rms_g: float, speed_kmh: float, base_g: float = 3.0) -> bool:
    """Return True when vibration RMS exceeds the speed-adjusted threshold."""
    return float(rms_g) > speed_threshold(speed_kmh, base_g=base_g)


def alert_level(rms_g: float, speed_kmh: float) -> str:
    """Classify vibration alert level as normal, warning or critical."""
    threshold = speed_threshold(speed_kmh)
    rms = float(rms_g)
    if rms > 1.5 * threshold:
        return "critical"
    if rms > threshold:
        return "warning"
    return "normal"

