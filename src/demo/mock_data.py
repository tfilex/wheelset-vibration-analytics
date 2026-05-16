from datetime import datetime, timedelta

import pandas as pd


SIGNAL_TYPES = (
    "Норма",
    "Дефект внутреннего кольца",
    "Дефект внешнего кольца",
)

TEST_BEARINGS = (
    "Bearing1_3",
    "Bearing1_4",
    "Bearing2_5",
    "Bearing3_3",
)

DASHBOARD_RUL = 34


def build_checks_history() -> pd.DataFrame:
    now = datetime.now().replace(second=0, microsecond=0)
    checks = [
        ("Букса 1", "WARNING", "34%"),
        ("Букса 2", "OK", "82%"),
        ("Букса 3", "OK", "76%"),
        ("Букса 4", "WARNING", "41%"),
        ("Букса 5", "OK", "69%"),
    ]

    return pd.DataFrame(
        [
            {
                "Время": (now - timedelta(minutes=4 * index)).strftime("%H:%M"),
                "Узел": node,
                "Статус": status,
                "RUL": rul,
            }
            for index, (node, status, rul) in enumerate(checks)
        ]
    )
