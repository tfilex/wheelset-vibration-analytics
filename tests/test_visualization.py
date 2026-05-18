import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.plots import (
    build_attribution_heatmap,
    build_gauge_figure,
    build_rul_figure,
    build_signal_figure,
)


def test_plot_builders_return_figures():
    signal = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "amplitude": [0.1, 0.2, 0.1],
        }
    )
    history = pd.DataFrame(
        {
            "step": [1, 2],
            "true_rul": [1.0, 0.8],
            "pred_rul": [0.9, 0.75],
        }
    )

    assert isinstance(build_signal_figure(signal), go.Figure)
    assert isinstance(build_rul_figure(history), go.Figure)
    assert isinstance(build_gauge_figure(34), go.Figure)
    assert isinstance(build_attribution_heatmap(np.ones((4, 4))), go.Figure)
